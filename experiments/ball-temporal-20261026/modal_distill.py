"""Bounded classifier-head fine-tuning from sequence verifier decisions."""
from pathlib import Path
import importlib.util
import modal
HERE=Path(__file__).parent
if modal.is_local():
    spec=importlib.util.spec_from_file_location('temporal_image',HERE/'modal_run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    image=(module.image.add_local_file(HERE/'distill.py','/work/distill.py')
           .add_local_file(HERE/'distill-protocol.json','/work/distill-protocol.json')
           .add_local_file(HERE/'distill-labels.json','/work/distill-labels.json')
           .add_local_file(HERE/'results/distill-transfer-inputs.json','/work/distill-transfer-inputs.json'))
else:image=modal.Image.debian_slim()
volume=modal.Volume.from_name('sport-scans-experiments')
app=modal.App('sport-scans-ball-verifier-distillation')

@app.function(image=image,cpu=8,memory=32768,gpu='H100',timeout=1800,scaledown_window=10,volumes={'/data':volume})
def run():
    import sys
    sys.path.insert(0,'/work')
    from distill import train
    volume.reload();return train(volume.commit)

@app.local_entrypoint()
def main():print(run.remote())
