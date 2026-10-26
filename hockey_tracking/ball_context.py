"""Bounded, offline ball inference in a camera-compensated rink plane.

Inputs are trusted *detector anchors*, not ground truth. Covariances describe
assumed localization and motion noise conditional on the ball being on the floor
and the anchors being correct. They are not calibrated probability guarantees.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    clear_gap_s: float = .4
    occluded_gap_s: float = .8
    clear_coast_s: float = .2
    occluded_coast_s: float = .4
    anchor_sigma_px: float = 6.
    acceleration_sigma: tuple = (.12, .30)
    max_speed: tuple = (.65, 1.6)
    max_radius_px: float = 100.
    use_occlusion: bool = True
    coast: bool = True
    bridge: bool = True


def project(H, p):
    q = np.asarray(H) @ np.r_[p, 1.]
    return q[:2] / q[2]


def jacobian(H, p):
    H = np.asarray(H)
    q = H @ np.r_[p, 1.]
    return (H[:2, :2] * q[2] - q[:2, None] * H[2, :2]) / q[2] ** 2


def normalized_homography(row, bounds):
    lo, hi, far, near = bounds
    N = np.array([[1/(hi-lo), 0, -lo/(hi-lo)],
                  [0, 1/(near-far), -far/(near-far)], [0, 0, 1]])
    return N @ np.asarray(row['image_to_floor'])


def occluders(row, pixel):
    """An image overlap is evidence of possible occlusion, never possession."""
    x, y = np.asarray(pixel) / [1920, 1080]
    return [o['track_id'] for o in row['objects']
            if o['bbox'][0] <= x <= o['bbox'][2]
            and o['bbox'][1] <= y <= o['bbox'][3]]


def infer(replay, anchors=None, config=Config()):
    rows, fps = replay['frames'], replay['fps']
    if anchors is None:
        anchors = {r['frame']: r['ball'] for r in rows if r.get('ball')}
    hs = {r['frame']: normalized_homography(r, replay['relative_bounds'])
          for r in rows if r['camera_valid'] and r.get('image_to_floor') is not None}
    anchors = {f: a for f, a in anchors.items() if f in hs}
    keys = sorted(anchors)
    positions = {f: np.array(a['xy'], float) for f, a in anchors.items()}
    covs = {f: (lambda J: J @ J.T * config.anchor_sigma_px**2)(
        jacobian(hs[f], np.array(anchors[f]['pixel']) * [1920, 1080])) for f in keys}
    out = [{'frame': f, 'state': 'unknown', 'reason':
            'no_supported_hypothesis' if f in hs else 'camera_unavailable'} for f in range(len(rows))]
    acceleration = np.diag(np.asarray(config.acceleration_sigma)**2)

    def emit(f, xy, cov, state, evidence, reason, age=0):
        if f not in hs or not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1):
            return False
        G = np.linalg.inv(hs[f]); pixel = project(G, xy)
        if not np.isfinite(pixel).all() or np.any(pixel < 0) or np.any(pixel > [1920, 1080]):
            return False
        J = jacobian(G, xy); pixel_cov = J @ cov @ J.T
        radius = 2 * np.sqrt(max(0., np.linalg.eigvalsh(pixel_cov)[-1]))
        if state != 'detected' and radius > config.max_radius_px:
            return False
        out[f] = dict(frame=f, state=state, reason=reason, xy=xy.tolist(),
                      pixel=(pixel/[1920, 1080]).tolist(), covariance=cov.tolist(),
                      pixel_covariance=pixel_cov.tolist(), model_radius_px=float(radius),
                      evidence_frames=evidence, nearest_evidence_s=age,
                      possible_occluders=occluders(rows[f], pixel))
        if state == 'detected': out[f]['detector_confidence'] = anchors[f]['confidence']
        return True

    for f in keys: emit(f, positions[f], covs[f], 'detected', [f], 'detector_anchor')

    def connected(a, b):
        return (0 < b-a <= round(config.occluded_gap_s*fps)
                and all(f in hs for f in range(a, b+1))
                and np.all(np.abs(positions[b]-positions[a]) <= np.asarray(config.max_speed)*(b-a)/fps))

    # Interpolation is an offline inference: both endpoints are required, and
    # ambiguity between local motion directions contributes to covariance.
    for i, (a, b) in enumerate(zip(keys, keys[1:])):
        if not config.bridge: break
        if b-a < 2 or not connected(a, b): continue
        dt = (b-a)/fps; v = (positions[b]-positions[a])/dt
        va, vb = v, v
        if i and connected(keys[i-1], a):
            va = (positions[a]-positions[keys[i-1]]) / ((a-keys[i-1])/fps)
        if i+2 < len(keys) and connected(b, keys[i+2]):
            vb = (positions[keys[i+2]]-positions[b]) / ((keys[i+2]-b)/fps)
        intermediate = []
        for f in range(a+1, b):
            w = (f-a)/(b-a); xy = (1-w)*positions[a]+w*positions[b]
            pixel = project(np.linalg.inv(hs[f]), xy)
            intermediate.append((f, w, xy, bool(occluders(rows[f], pixel))))
        occluded = config.use_occlusion and sum(x[3] for x in intermediate) >= len(intermediate)/2
        if dt > (config.occluded_gap_s if occluded else config.clear_gap_s) + 1e-9: continue
        for f, w, xy, _ in intermediate:
            forward = positions[a] + va*w*dt
            backward = positions[b] - vb*(1-w)*dt
            disagreement = forward-backward
            cov = ((1-w)*covs[a]+w*covs[b]
                   + acceleration*(dt*dt*w*(1-w))**2
                   + w*(1-w)*np.outer(disagreement, disagreement))
            emit(f, xy, cov, 'inferred', [a, b],
                 'occlusion_bridge' if occluded else 'short_gap_bridge', min(f-a,b-f)/fps)

    if config.coast:
        for i, a in enumerate(keys):
            if not i or not connected(keys[i-1], a): continue
            previous = keys[i-1]; dt = (a-previous)/fps
            velocity = (positions[a]-positions[previous])/dt
            # A contradictory forthcoming anchor prevents a one-sided guess.
            following = keys[i+1] if i+1 < len(keys) else None
            if following is not None and following-a <= round(config.occluded_gap_s*fps) and not connected(a, following):
                continue
            for f in range(a+1, min(len(rows), a+round(config.occluded_coast_s*fps)+1)):
                if f not in hs: break
                if following is not None and f >= following: break
                if out[f]['state'] != 'unknown': continue
                age = (f-a)/fps; xy = positions[a]+velocity*age
                pixel = project(np.linalg.inv(hs[f]), xy)
                blocked = config.use_occlusion and bool(occluders(rows[f], pixel))
                if age > (config.occluded_coast_s if blocked else config.clear_coast_s)+1e-9: break
                # Position/velocity correlation matters when the velocity is a
                # finite difference of the same noisy pair of observations.
                ratio = age/dt
                cov = ((1+ratio)**2*covs[a]+ratio**2*covs[previous]
                       + acceleration*age**4/4)
                if not emit(f, xy, cov, 'inferred', [previous, a],
                            'occlusion_prediction' if blocked else 'motion_prediction', age): break
    return {'schema_version': 1, 'fps': fps, 'duration_s': replay['duration_s'],
            'source_id': replay['source_id'], 'start_s': replay['start_s'],
            'config': asdict(config), 'uncertainty_calibrated': False,
            'floor_projection_assumption': True, 'frames': out}
