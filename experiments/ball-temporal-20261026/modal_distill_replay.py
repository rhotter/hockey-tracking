"""Run the bounded replay-training follow-up on Modal."""
from pathlib import Path
import importlib.util
import json
import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else Path('/work')
app = modal.App('sport-scans-ball-verifier-replay-training')
volume = modal.Volume.from_name('sport-scans-experiments')
if modal.is_local():
    spec = importlib.util.spec_from_file_location('distill_image', HERE/'modal_distill.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    image = (module.image.add_local_file(HERE/'distill_replay.py', '/work/distill_replay.py')
             .add_local_file(HERE/'distill-replay-protocol.json', '/work/distill-replay-protocol.json'))
else:
    image = modal.Image.debian_slim()


@app.function(image=image, cpu=8, memory=32768, gpu='H100', timeout=1800,
              scaledown_window=10, volumes={'/data': volume})
def run():
    import sys
    sys.path.insert(0, '/work')
    from distill_replay import train
    volume.reload(); return train(volume.commit)


@app.local_entrypoint()
def main():
    report = run.remote()
    (ROOT/'build/ball-temporal-20261026/distillation-replay-report.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print({name: {'test': row['test'], 'selected_epoch': row['selected_epoch']} for name, row in report['variants'].items()})
