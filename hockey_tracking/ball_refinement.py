"""Sequence sightings, independently restarted masks, and one output per frame.

No review labels enter this fusion. A sequence-model sighting must be supported
by a localized detector or mutually consistent SAM tracks. Between sightings,
tracks from distinct seed frames must agree; a repeated vote from one seed does
not count as independent temporal support. Scores are not probabilities.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import numpy as np
from .ball_context import normalized_homography, project
from .ball_sequence import camera_warp


@dataclass(frozen=True)
class RefinementConfig:
    seed_mask_error_px: float = 12.
    seed_detector_radius_px: float = 12.
    detector_score: float = .05
    detector_appearance: float = .1
    cycle_error_px: float = 10.
    cycle_gap_frames: int = 6
    track_agreement_px: float = 10.
    single_track_seed_distance: int = 2
    maximum_interpolation_gap: int = 12
    maximum_camera_corrected_speed_px: float = 140.
    preserve_visual_detections: bool = True
    minimum_output_appearance: float = 0.
    held_as_context: bool = False
    prefer_appearance_center: bool = False
    reject_isolated_spikes: bool = False


def valid_mask(row):
    return bool(row and row.get('visible') and row.get('shape_valid') and
                row.get('inside_image') and row.get('inside_crop') and
                len(row.get('pixel', [])) == 2 and
                all(math.isfinite(x) for x in row['pixel']))


def detector_support(frame, pixel, config):
    nearby = [p for p in frame.get('candidates', [])
              if p['score'] >= config.detector_score and
              p.get('appearance_score', 0.) >= config.detector_appearance and
              math.dist(p['pixel'], pixel) <= config.seed_detector_radius_px]
    return max(nearby, key=lambda p:p['score']*p['appearance_score']) if nearby else None


def mask_clusters(proposals, radius):
    """One vote per seed frame, even if multiple models reuse that sighting."""
    clusters = []
    for proposal in sorted(proposals, key=lambda p:(p['seed_distance'], -p['appearance_score'])):
        cluster = next((c for c in clusters if math.dist(c['pixel'], proposal['pixel']) <= radius), None)
        if cluster is None:
            clusters.append({'pixel': proposal['pixel'], 'members': {proposal['seed_frame']: proposal}})
        elif proposal['seed_frame'] not in cluster['members']:
            cluster['members'][proposal['seed_frame']] = proposal
    return clusters


def make_position(rink, frame, pixel, support, state, reason, **evidence):
    if not all(math.isfinite(x) for x in pixel) or not (0 <= pixel[0] < 1920 and 0 <= pixel[1] < 1080):
        raise ValueError('Position outside source image')
    xy = None
    row = rink['frames'][frame]
    if support == 'floor' and row.get('camera_valid') and row.get('image_to_floor') is not None:
        floor = project(normalized_homography(row, rink['relative_bounds']), np.array(pixel))
        if np.isfinite(floor).all() and np.all(floor >= 0) and np.all(floor <= 1): xy = floor.tolist()
    return {'frame': frame, 'state': state, 'pixel': [pixel[0]/1920,pixel[1]/1080],
            'xy': xy, 'review_support': support, 'reason': reason, **evidence}


def refine_sequence(rink, baseline, seed_streams, track_streams, detector_frames,
                    config=RefinementConfig(), excluded_frames=()):
    n = len(rink['frames']); excluded = set(excluded_frames)
    if len(baseline['frames']) != n or len(detector_frames) != n:
        raise ValueError('Mismatched frame counts')
    seeds, tracks = {}, {}
    for data in seed_streams:
        for seed in data['seeds']:
            if not isinstance(seed['frame'],int) or not 0 <= seed['frame'] < n:
                raise ValueError('Invalid seed frame')
            key = (data['batch'], seed['frame'])
            if key in seeds: raise ValueError('Duplicate seed key')
            seeds[key] = seed
    seen_tracks = set()
    for data in track_streams:
        if not data.get('complete'): raise ValueError('Incomplete tracker output')
        if data['protocol'].get('offload_state_to_cpu', True):
            raise ValueError('Only validated GPU-state tracks may be promoted')
        for track in data['tracks']:
            key = (data['protocol']['batch'], track['seed_frame'])
            if key in seen_tracks: raise ValueError('Duplicate tracking seed')
            seen_tracks.add(key)
            if key not in seeds: raise ValueError('Unknown tracking seed')
            if math.dist(track['seed_pixel'], seeds[key]['pixel']) > 1e-6:
                raise ValueError('Tracking seed coordinates changed')
            rows = {r['frame']:r for r in track['rows']}
            if any(not isinstance(f,int) or not 0 <= f < n for f in rows):
                raise ValueError('Invalid tracking frame')
            if len(rows) != len(track['rows']): raise ValueError('Duplicate tracking frame')
            own = rows.get(track['seed_frame'])
            if valid_mask(own) and math.dist(own['pixel'], track['seed_pixel']) <= config.seed_mask_error_px:
                tracks[key] = {'meta':track, 'rows':rows}
    partners = {key:set() for key in tracks}
    ordered = sorted(tracks, key=lambda key:key[1])
    for index, a in enumerate(ordered):
        for b in ordered[index+1:]:
            af,bf = a[1],b[1]
            if bf-af > config.cycle_gap_frames: break
            if af == bf or any(f in excluded for f in range(af,bf+1)): continue
            forward,backward = tracks[a]['rows'].get(bf),tracks[b]['rows'].get(af)
            if (valid_mask(forward) and valid_mask(backward) and
                math.dist(forward['pixel'],seeds[b]['pixel']) <= config.cycle_error_px and
                math.dist(backward['pixel'],seeds[a]['pixel']) <= config.cycle_error_px):
                partners[a].add(b);partners[b].add(a)
    accepted, seed_details = {}, []
    for key, seed in seeds.items():
        f = key[1]
        cnn = detector_support(detector_frames[f], seed['pixel'], config)
        mask_ok = key in tracks
        cycle = bool(partners.get(key))
        accept = f not in excluded and mask_ok and bool(cnn or cycle)
        seed_details.append({'batch':key[0], 'frame':f, 'accepted':accept,
                             'seed_mask_valid':mask_ok, 'detector_agreement':cnn is not None,
                             'cycle_partners':[list(p) for p in sorted(partners.get(key, set()))]})
        if accept: accepted[key] = {'seed':seed, 'detector':cnn, 'cycle':cycle}
    proposals = [[] for _ in range(n)]
    for key, info in accepted.items():
        for f, row in tracks[key]['rows'].items():
            if f in excluded or not valid_mask(row): continue
            if any(c in excluded for c in range(min(f,key[1]),max(f,key[1])+1)): continue
            proposals[f].append({**row, 'seed_frame':key[1], 'seed_key':key,
                                'seed_distance':abs(f-key[1]), 'support':info['seed']['support']})
    result = deepcopy(baseline); changed = {}; direct = {}
    for key, info in accepted.items():
        f = key[1]
        rank = (info['cycle'], info['detector']['score'] if info['detector'] else 0.,
                info['seed']['appearance_score'])
        if f not in direct or rank > direct[f][0]: direct[f] = (rank,key,info)
    for f in range(n):
        if f in excluded:
            result['frames'][f] = {'frame':f,'state':'unknown','reason':'edit_or_unavailable_context'}
            continue
        if config.preserve_visual_detections and baseline['frames'][f]['state'] == 'detected': continue
        if f in direct and tracks[direct[f][1]]['rows'][f]['appearance_score'] >= config.minimum_output_appearance:
            _,key,info = direct[f]; row = tracks[key]['rows'][f]
            state = 'context' if config.held_as_context and info['seed']['support']=='held' else 'detected'
            detector_seed = info['seed'].get('source') == 'detector_consensus'
            result['frames'][f] = make_position(rink,f,row['pixel'],info['seed']['support'],state,
                'detector_consensus_with_sam_corroboration' if detector_seed else 'sequence_sighting_with_sam_corroboration', evidence=info['seed']['evidence'],
                evidence_frames=sorted({f}|{p[1] for p in partners[key]}), tracker='sam3_gpu_state',
                appearance_score=row['appearance_score'], detector_agreement=info['detector'] is not None)
            changed[f] = 'detector_consensus_sighting' if detector_seed else 'sequence_sighting'
            continue
        clusters = mask_clusters(proposals[f], config.track_agreement_px)
        eligible = []
        for cluster in clusters:
            members = list(cluster['members'].values()); seed_frames = sorted(cluster['members'])
            bracketed = len(seed_frames) >= 2 and seed_frames[0] < f < seed_frames[-1]
            near = min(p['seed_distance'] for p in members) <= config.single_track_seed_distance
            cnn = detector_support(detector_frames[f], cluster['pixel'], config)
            if bracketed or (near and cnn):
                eligible.append((bracketed,len(seed_frames),max(p['appearance_score'] for p in members),cluster))
        if eligible:
            bracketed,_,_,cluster = max(eligible,key=lambda c:c[:3])
            members = list(cluster['members'].values())
            chosen = (max(members,key=lambda p:(p['appearance_score'],-p['seed_distance']))
                      if config.prefer_appearance_center else
                      min(members,key=lambda p:(p['seed_distance'],-p['appearance_score'])))
            if chosen['appearance_score'] < config.minimum_output_appearance: continue
            supports = {p['support'] for p in members}; support = supports.pop() if len(supports)==1 else 'unclear'
            state = 'context' if config.held_as_context and support=='held' else 'tracked'
            result['frames'][f] = make_position(rink,f,chosen['pixel'],support,state,
                'sam_tracks_agree_between_sightings' if bracketed else 'short_sam_track_with_detector',
                tracker='sam3_gpu_state', evidence_frames=sorted(cluster['members']),
                appearance_score=chosen['appearance_score'], temporal_votes=len(cluster['members']),
                evidence='SAM tracks restarted from distinct ball sightings agree here.' if bracketed else
                         'A recent sequence sighting, SAM mask, and local detector agree here.')
            changed[f] = 'mask_consensus' if bracketed else 'mask_detector'
    # A corroborated detector may still latch onto a second pink object. Reject
    # a new one-frame excursion only when observations on both sides agree on
    # a plausible path. Context guesses cannot veto a visual observation.
    isolated_rejections = []
    if config.reject_isolated_spikes:
        for f in sorted(changed):
            if not 0 < f < n-1: continue
            a, middle, b = result['frames'][f-1:f+2]
            if any(r['state'] not in ('detected','tracked') for r in (a,middle,b)): continue
            if any(not rink['frames'][j].get('camera_valid') or
                   rink['frames'][j].get('image_to_floor') is None for j in (f-1,f,f+1)): continue
            ap,mp,bp = [np.array(r['pixel'])*[1920,1080] for r in (a,middle,b)]
            incoming = float(np.linalg.norm(camera_warp(rink,f-1,f,ap)-mp))
            outgoing = float(np.linalg.norm(camera_warp(rink,f,f+1,mp)-bp))
            bridge = float(np.linalg.norm(camera_warp(rink,f-1,f+1,ap)-bp)/2)
            limit = config.maximum_camera_corrected_speed_px
            if incoming > limit and outgoing > limit and bridge <= limit:
                isolated_rejections.append({'frame':f,'pixel':middle['pixel'],
                    'incoming_px':incoming,'outgoing_px':outgoing,'bridge_px_per_frame':bridge})
        for rejection in isolated_rejections:
            f=rejection['frame']
            result['frames'][f]={'frame':f,'state':'unknown','reason':'isolated_observation_contradicts_neighbors'}
            changed[f]='isolated_rejection'
    # Interpolation remains explicitly inferred, including when it replaces an
    # older possession guess that stayed on the passer after the release.
    observations = [f for f,r in enumerate(result['frames']) if r['state'] in ('detected','tracked')]
    for af,bf in zip(observations,observations[1:]):
        if bf-af <= 1 or bf-af > config.maximum_interpolation_gap: continue
        if not any(f in changed for f in (af,bf)) and not any(
                af < rejection['frame'] < bf for rejection in isolated_rejections): continue
        if any(f in excluded for f in range(af,bf+1)): continue
        # A supported held-ball position must not be replaced by a straight path.
        if any(f in changed and result['frames'][f].get('review_support')=='held'
               for f in range(af+1,bf)): continue
        a,b = result['frames'][af],result['frames'][bf]
        ap,bp = np.array(a['pixel'])*[1920,1080],np.array(b['pixel'])*[1920,1080]
        speed = np.linalg.norm(camera_warp(rink,af,bf,ap)-bp)/(bf-af)
        if speed > config.maximum_camera_corrected_speed_px: continue
        support = a.get('review_support','unclear') if a.get('review_support')==b.get('review_support') else 'unclear'
        for f in range(af+1,bf):
            u=(f-af)/(bf-af); pixel=(1-u)*camera_warp(rink,af,f,ap)+u*camera_warp(rink,bf,f,bp)
            if not (np.isfinite(pixel).all() and np.all(pixel>=0) and np.all(pixel<[1920,1080])): continue
            result['frames'][f] = make_position(rink,f,pixel,support,'inferred','short_gap_between_verified_tracks',
                evidence_frames=[af,bf],evidence='Short camera-compensated interpolation between corroborated ball positions.')
            changed[f] = 'short_gap'
    result.update(method='Sequence ball sightings, GPU-state SAM reseeding, temporal agreement, and detector corroboration',
                  refinement_config=asdict(config), independent_human_ground_truth=False)
    diagnostics = {'config':asdict(config), 'seed_count':len(seeds), 'valid_seed_masks':len(tracks),
                   'accepted_seeds':len(accepted), 'seed_details':seed_details,
                   'changes_by_source':dict(Counter(changed.values())),
                   'before_states':dict(Counter(r['state'] for r in baseline['frames'])),
                   'after_states':dict(Counter(r['state'] for r in result['frames'])),
                   'changed_frames':sorted(changed),
                   'isolated_observation_rejections':isolated_rejections,
                   'limits':'No review labels enter fusion. Corroboration is not calibrated confidence; inferred frames are not visual detections.'}
    return result,diagnostics
