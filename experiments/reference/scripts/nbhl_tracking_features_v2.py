"""Appearance extraction and scene-boundary auditing on fixed NBHL detections."""
import copy
import json
from pathlib import Path
from nbhl_tracking_fourhour import sha, dump


def frame_folder(base, meta):
    if meta['long']:
        return base/'long-sam2-v1'/meta['summary']['source_id']/'frames'
    return Path('/data/sequences')/meta['summary']['clip']/'img1'


def cut_audit(base, out, volume):
    import cv2
    import numpy as np
    from nbhl_tracking_fourhour import load_development
    reports = []
    cv2.setNumThreads(2)
    for meta, _ in load_development(base, 'generic'):
        clip = meta['summary']['clip']
        folder = frame_folder(base, meta)
        paths = sorted(folder.glob('*.jpg'))
        assert len(paths) == len(meta['frames']), (clip, len(paths))
        previous = None
        changes = []
        for index, path in enumerate(paths):
            pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            assert pixels is not None
            gray = cv2.resize(pixels, (160, 90))
            change = float(np.mean(cv2.absdiff(gray, previous))) if previous is not None else 0.
            changes.append(change)
            previous = gray
        cuts = [i for i, value in enumerate(changes) if value > 55.]
        report = {'clip': clip, 'reset_frames': cuts, 'criterion': 'Mean absolute grayscale difference >55 at 160x90; same threshold as existing long-clip pipeline.',
                  'changes': changes, 'input_sha256': meta['summary']['input_sha256'], 'code_sha256': sha(__file__)}
        dump(out/'cuts'/(clip+'.json'), report)
        reports.append({k: v for k, v in report.items() if k != 'changes'})
    volume.commit()
    return reports


def apply_cuts(meta, features, cut_report):
    import numpy as np
    meta, features = copy.deepcopy(meta), dict(features)
    assert cut_report['input_sha256'] == meta['summary']['input_sha256']
    resets = np.zeros(len(meta['frames']), dtype=bool)
    resets[cut_report['reset_frames']] = True
    resets |= features['resets']
    features['resets'] = resets
    features['warps'] = features['warps'].copy()
    features['warps'][resets] = np.eye(2, 3)
    scene = 0
    for i, row in enumerate(meta['frames']):
        scene += bool(resets[i])
        row['scene'] = scene
    meta['scene_cut_policy'] = cut_report['criterion']
    return meta, features


def extract(base, out, variant, generic, volume):
    import time
    import numpy as np
    import torch
    from PIL import Image
    from nbhl_identity_contrastive import load_model, preprocess
    from nbhl_tracking_fourhour import load_development
    torch.set_num_threads(4)
    name = variant + ('_generic' if generic else '')
    checkpoint = None if generic else out/'contrastive'/variant/'best.pt'
    if checkpoint:
        assert sha(checkpoint) == json.loads((checkpoint.parent/'completed.json').read_text())['weights_sha256']
    model, provenance = load_model(base, variant, checkpoint)
    transform = preprocess(variant)
    dim = 1536 if variant.startswith('osnet_parts') else 512 if variant.startswith('osnet') else 768
    reports = []
    for meta, cached in load_development(base, 'generic'):
        clip = meta['summary']['clip']
        dest = out/'features'/name/clip
        if (dest/'metadata.json').exists():
            reports.append(json.loads((dest/'metadata.json').read_text())['summary'])
            continue
        dest.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        folder = frame_folder(base, meta)
        paths = sorted(folder.glob('*.jpg'))
        assert len(paths) == len(meta['frames'])
        features = np.zeros((int(cached['offsets'][-1]), dim), np.float32)
        batch, indices = [], []
        def flush():
            if not batch:
                return
            with torch.inference_mode():
                z = model(torch.stack(batch).cuda()).float().cpu().numpy()
            assert np.isfinite(z).all() and np.allclose(np.linalg.norm(z, axis=1), 1., atol=1e-4)
            features[indices] = z
            batch.clear()
            indices.clear()
        for f, row in enumerate(meta['frames']):
            im = Image.open(paths[f]).convert('RGB')
            w, h = im.size
            for j, d in enumerate(row['detections']):
                if d[4] < meta['config']['track_low_thresh']:
                    continue
                box = (max(0., d[0]), max(0., d[1]), min(w, d[2]), min(h, d[3]))
                assert box[2] > box[0] and box[3] > box[1]
                batch.append(transform(im.crop(box)))
                indices.append(int(cached['offsets'][f])+j)
                if len(batch) == 32:
                    flush()
        flush()
        cached['features'] = features
        np.savez_compressed(dest/'features.npz', **cached)
        meta['summary'].update(variant=name, feature_sha256=sha(dest/'features.npz'), seconds=time.monotonic()-started,
                               weights_sha256=sha(checkpoint) if checkpoint else None, base_model=provenance,
                               feature_dimensions=dim, code_sha256=sha(__file__))
        dump(dest/'metadata.json', meta)
        reports.append(meta['summary'])
        volume.commit()
        print(json.dumps(meta['summary']), flush=True)
    return reports


def load_features(base, out, variant, cuts=False):
    import numpy as np
    from nbhl_tracking_fourhour import load_development
    if variant in ['none', 'generic', 'extended_full']:
        rows = load_development(base, variant)
    else:
        rows = []
        for path in sorted((out/'features'/variant).glob('*/metadata.json')):
            meta = json.loads(path.read_text())
            npz = path.with_name('features.npz')
            assert sha(npz) == meta['summary']['feature_sha256']
            rows.append((meta, dict(np.load(npz))))
        assert len(rows) == 9, (variant, len(rows))
    if cuts:
        rows = [apply_cuts(meta, features, json.loads((out/'cuts'/(meta['summary']['clip']+'.json')).read_text()))
                for meta, features in rows]
    return rows
