"""Export camera-compensated neighboring detector candidates without labels."""
from pathlib import Path
import importlib.util
import json
import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else Path('/work')
app = modal.App('sport-scans-ball-motion-selection')
volume = modal.Volume.from_name('sport-scans-experiments')
if modal.is_local():
    spec = importlib.util.spec_from_file_location('temporal_image', HERE/'modal_run.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    image = module.image.add_local_file(HERE/'motion-selection-protocol.json', '/work/motion-selection-protocol.json')
else:
    image = modal.Image.debian_slim()


@app.function(image=image, cpu=8, memory=16384, timeout=1200, volumes={'/data': volume})
def run():
    import sys
    sys.path.insert(0, '/work')
    import cv2
    import numpy as np
    from worker import BASE, OUT, registration, read, write, sha
    cv2.setNumThreads(2); volume.reload()
    protocol = read('/work/motion-selection-protocol.json')
    destination = OUT/'motion-selection-inputs.json'
    if destination.exists(): return read(destination)
    rows = []
    for clip in protocol['clips']:
        source = (BASE/'new-games-20261025-v1/frames'/clip if clip.endswith('0600')
                  else OUT/'extra-frames'/clip)
        detector_path = OUT/'yolo-full1920'/f'{clip}.json'
        detector = read(detector_path); assert detector['complete']
        by_frame = {r['frame']: r for r in detector['frames']}
        for f in protocol['sample_frames']:
            target_path = source/f'{f+1:06d}.jpg'
            assert sha(target_path) == by_frame[f]['source_sha256']
            target = cv2.cvtColor(cv2.imread(str(target_path)), cv2.COLOR_BGR2RGB)
            frames = []
            for offset in protocol['offsets']:
                row = by_frame[f+offset]
                path = source/f'{f+offset+1:06d}.jpg'
                assert sha(path) == row['source_sha256']
                if offset:
                    raw = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                    matrix, quality = registration(raw, target)
                else:
                    matrix, quality = np.eye(2, 3), 1.
                frames.append({**row, 'offset': offset, 'to_target': matrix.tolist(), 'registration_quality': quality})
            rows.append({'clip': clip, 'frame': f, 'detector_sha256': sha(detector_path), 'frames': frames})
        print({'clip': clip, 'completed_targets': len(rows)}, flush=True)
    result = {'protocol': protocol, 'rows': rows}
    write(destination, result); volume.commit(); return result


@app.local_entrypoint()
def main():
    result = run.remote()
    destination = ROOT/'build/ball-temporal-20261026/motion-selection-inputs.json'
    destination.write_text(json.dumps(result, separators=(',', ':'))+'\n')
    print({'targets': len(result['rows'])})
