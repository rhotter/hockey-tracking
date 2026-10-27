"""Score every cached YOLO proposal with the completed replay-trained verifiers."""
from pathlib import Path
import importlib.util
import json
import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else Path('/work')
app = modal.App('sport-scans-ball-verifier-reranking')
volume = modal.Volume.from_name('sport-scans-experiments')
if modal.is_local():
    spec = importlib.util.spec_from_file_location('distill_image', HERE/'modal_distill.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    image = (module.image.add_local_file(HERE/'verifier-rerank-protocol.json', '/work/verifier-rerank-protocol.json')
             .add_local_file(HERE/'multiview-transfer-protocol.json', '/work/multiview-transfer-protocol.json')
             .add_local_file(ROOT/'hockey_tracking/ball_path.py', '/work/ball_path.py'))
else:
    image = modal.Image.debian_slim()


@app.function(image=image, cpu=8, memory=16384, gpu='H100', timeout=900,
              scaledown_window=10, volumes={'/data': volume})
def run(multiview: bool = False):
    import sys
    sys.path.insert(0, '/work')
    import cv2
    import numpy as np
    import torch
    from crop_verifier import SpatialBallVerifier, patch_inputs
    from worker import read, write, sha, BASE, OUT
    torch.set_num_threads(4); cv2.setNumThreads(1); volume.reload()
    protocol = read('/work/multiview-transfer-protocol.json' if multiview else '/work/verifier-rerank-protocol.json')
    if multiview:
        from ultralytics import YOLO
        from ball_path import fuse_candidates
        detector_model = YOLO(str(BASE/'ball/training/yolo11m-crops640-s17-f10-e80-c6ab3e44/weights/best.pt'))
    models, weights, outputs = {}, {}, {}
    for name in protocol['variants']:
        path = BASE/'ball-rebuild-20261026-v1/verifier-v4/best.pt' if name == 'original' else OUT/'distillation-replay'/f'{name}.pt'
        model = SpatialBallVerifier().cuda().eval()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['state_dict'])
        models[name] = model; weights[name] = sha(path); outputs[name] = []
    for clip in protocol['clips']:
        source = BASE/'new-games-20261025-v1/frames'/clip if clip.endswith('0600') else OUT/'extra-frames'/clip
        path = OUT/'yolo-full1920'/f'{clip}.json'; detector = read(path); assert detector['complete']
        by_frame = {r['frame']: r for r in detector['frames']}
        frames = protocol['sample_frames_last_two'] if clip.endswith('1800') else protocol['sample_frames_first_four']
        for f in frames:
            row = by_frame[f]; image_path = source/f'{f+1:06d}.jpg'; assert sha(image_path) == row['source_sha256']
            image_rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            candidates = row['candidates']
            if multiview:
                bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                height, width = bgr.shape[:2]
                streams = []
                for size in protocol['views']['full']:
                    prediction = detector_model.predict(bgr, imgsz=size, conf=.015, iou=.4, device=0, max_det=25, verbose=False)[0]
                    streams.append([{'detections': prediction.boxes.data.cpu().tolist()}])
                def origins(extent): return sorted(set(list(range(0, max(1, extent-640), 480))+[max(0, extent-640)]))
                offsets = [(x, y) for y in origins(height) for x in origins(width)]
                tile_predictions = detector_model.predict([bgr[y:y+640, x:x+640] for x, y in offsets],
                    imgsz=640, conf=.015, iou=.4, device=0, max_det=25, verbose=False)
                boxes = []
                for prediction, (x, y) in zip(tile_predictions, offsets):
                    for box in prediction.boxes.data.cpu().numpy():
                        box[:4] += np.array([x, y, x, y]); boxes.append(box.tolist())
                streams.append([{'detections': boxes}])
                boxes = fuse_candidates(streams)[0]['detections']
                candidates = [{'pixel': ((np.array(b[:2])+b[2:4])/2).tolist(), 'score': b[4]} for b in boxes]
            x = torch.from_numpy(np.stack([patch_inputs(image_rgb, c['pixel']) for c in candidates])).cuda() if candidates else None
            for name, model in models.items():
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    scores = model(x).float().sigmoid().cpu().tolist() if x is not None else []
                outputs[name].append({**row, 'detector_sha256': sha(path),
                                      'candidates': [{**c, 'appearance_score': s} for c, s in zip(candidates, scores)]})
        print({'clip': clip, 'frames': len(frames)}, flush=True)
    report = {'protocol': protocol, 'models': {name: {'weights_sha256': weights[name], 'frames': frames} for name, frames in outputs.items()}}
    write(OUT/('multiview-transfer.json' if multiview else 'verifier-rerank.json'), report); volume.commit(); return report


@app.local_entrypoint()
def main(multiview: bool = False):
    report = run.remote(multiview)
    destination = ROOT/'build/ball-temporal-20261026'/('multiview-transfer.json' if multiview else 'verifier-rerank.json')
    destination.write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print({'models': list(report['models']), 'frames_each': len(next(iter(report['models'].values()))['frames'])})
