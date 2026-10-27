"""Test pretrained SAM3 playing-surface masks as ball-candidate context."""
from pathlib import Path
import importlib.util
import json
import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else Path('/work')
app = modal.App('sport-scans-ball-playing-surface')
volume = modal.Volume.from_name('sport-scans-experiments')
if modal.is_local():
    spec = importlib.util.spec_from_file_location('sam_image', ROOT/'experiments/ball-sam-20261026/modal_run.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    image = module.image.add_local_file(HERE/'surface-filter-protocol.json', '/work/surface-filter-protocol.json')
else:
    image = modal.Image.debian_slim()


@app.function(image=image, cpu=8, memory=32768, gpu='H100', timeout=1200,
              scaledown_window=10, volumes={'/data': volume})
def run():
    import hashlib
    import time
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    volume.reload(); torch.set_num_threads(4); cv2.setNumThreads(2)
    base = Path('/data/ball-hockey/reliability-20261018'); out = base/'ball-temporal-20261026-v1'
    protocol = json.loads(Path('/work/surface-filter-protocol.json').read_text())
    source = out/'verifier-rerank.json'; candidates = json.loads(source.read_text())
    weights = hf_hub_download('facebook/sam3', 'sam3.pt', revision='3c879f39826c281e95690f02c7821c4de09afae7', local_files_only=True)
    model = build_sam3_image_model(checkpoint_path=weights, load_from_HF=False).eval()
    processor = Sam3Processor(model, confidence_threshold=protocol['mask_confidence'])
    rows = []; started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i, row in enumerate(candidates['models']['original']['frames']):
            clip = row['clip']; f = row['frame']
            folder = base/'new-games-20261025-v1/frames'/clip if clip.endswith('0600') else out/'extra-frames'/clip
            path = folder/f'{f+1:06d}.jpg'; assert hashlib.sha256(path.read_bytes()).hexdigest() == row['source_sha256']
            raw = Image.open(path).convert('RGB'); width, height = raw.size
            state = processor.set_image(raw)
            result = processor.set_text_prompt(state=state, prompt=protocol['prompt'])
            scores = result['scores'].float().cpu().numpy()
            masks = result['masks'].detach().cpu().numpy().reshape(-1, height, width)
            chosen = masks[scores >= protocol['mask_confidence']]
            union = np.any(chosen, axis=0) if len(chosen) else np.zeros((height, width), bool)
            small = cv2.resize(union.astype(np.uint8), tuple(protocol['mask_size']), interpolation=cv2.INTER_NEAREST)
            fraction = float(small.mean()); active = protocol['minimum_fraction'] <= fraction <= protocol['maximum_fraction']
            yy, xx = np.nonzero(small)
            hull = cv2.convexHull(np.stack([xx, yy], 1).astype(np.float32)).reshape(-1, 2) if active else np.empty((0, 2))
            hull = hull * [width/protocol['mask_size'][0], height/protocol['mask_size'][1]]
            rows.append({'clip': clip, 'frame': f, 'source_sha256': row['source_sha256'],
                         'width': width, 'height': height, 'surface_fraction': fraction,
                         'active': active, 'mask_scores': scores.tolist(), 'hull': hull.tolist()})
            if i % 15 == 0: print({'frames': i+1, 'seconds': time.monotonic()-started}, flush=True)
    report = {'protocol': protocol, 'candidate_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'weights_sha256': hashlib.sha256(Path(weights).read_bytes()).hexdigest(), 'rows': rows,
              'seconds': time.monotonic()-started}
    (out/'surface-filter.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    volume.commit(); return report


@app.local_entrypoint()
def main():
    report = run.remote()
    (ROOT/'build/ball-temporal-20261026/surface-filter.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print({'frames': len(report['rows']), 'active_masks': sum(r['active'] for r in report['rows']), 'seconds': report['seconds']})
