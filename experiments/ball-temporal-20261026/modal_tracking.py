"""Short, repeatedly seeded tracking from sequence-localized ball sightings."""
from pathlib import Path
import importlib.util
import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else Path('/work')


def load_image(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.image


app = modal.App('sport-scans-ball-short-tracking')
volume = modal.Volume.from_name('sport-scans-experiments')
tap_image = load_image('point_image', 'experiments/ball-sequence-20261026/modal_points.py') if modal.is_local() else modal.Image.debian_slim()
sam_image = load_image('sam_image', 'experiments/ball-sam-20261026/modal_run.py') if modal.is_local() else modal.Image.debian_slim()


def extend(image):
    if not modal.is_local(): return image
    return (image.add_local_file(HERE/'dense_tracking.py', '/work/dense_tracking.py')
            .add_local_file(HERE/'refinement-pilot-audit.json', '/work/pilot-audit.json')
            .add_local_file(ROOT/'experiments/ball-rebuild-20261026/crop_verifier.py', '/work/crop_verifier.py'))


@app.function(image=extend(tap_image), cpu=8, memory=32768, gpu='H100',
              timeout=3600, scaledown_window=10, volumes={'/data': volume})
def prepare(batch: str, pilot: bool):
    import sys
    sys.path.insert(0, '/work')
    from dense_tracking import prepare_seeds
    volume.reload()
    return prepare_seeds(batch, pilot, volume.commit)


@app.function(image=extend(tap_image), cpu=8, memory=32768, gpu='H100',
              timeout=7200, max_containers=2, scaledown_window=10, volumes={'/data': volume})
def tap(batch: str):
    import sys
    sys.path.insert(0, '/work')
    from dense_tracking import run_tracking
    volume.reload()
    return run_tracking('tapnextpp', batch, volume.commit)


@app.function(image=extend(sam_image), cpu=8, memory=98304, gpu='H100',
              timeout=7200, max_containers=2, scaledown_window=10, volumes={'/data': volume})
def sam(batch: str, kind: str='sam3'):
    import sys
    sys.path.insert(0, '/work')
    from dense_tracking import run_tracking
    volume.reload()
    if batch.startswith('transfer-'):
        from dense_tracking import prepare_transfer
        assert prepare_transfer(batch[len('transfer-'):-len('-gpu')],volume.commit) == batch
    if batch.startswith('dense-wide-') or batch in ['dense-close-gpu','dense-motion-gpu']:
        from dense_tracking import prepare_crop_variant
        prepare_crop_variant(batch,volume.commit)
    return run_tracking(kind, batch, volume.commit)


@app.local_entrypoint()
def main(task: str='prepare', batch: str='pilot', pilot: bool=True):
    if task == 'prepare': print(prepare.remote(batch, pilot))
    elif task == 'tap': print(tap.remote(batch))
    elif task == 'sam': print(sam.remote(batch))
    elif task == 'sam-point': print(sam.remote(batch,'sam3-point'))
    elif task == 'sam2': print(sam.remote(batch,'sam2-box10'))
    elif task == 'sam-crops':
        print(list(sam.starmap([(name,'sam3-box6') for name in ['dense-wide-gpu','dense-close-gpu','dense-motion-gpu']])))
    elif task == 'sam-masks':
        print(list(sam.starmap([(batch,kind) for kind in ['sam3-mask3','sam3-mask5']])))
    elif task == 'sam-prompts':
        print(list(sam.starmap([(batch,kind) for kind in ['sam3-box6','sam3-box14','sam3-point-negative']])))
    elif task == 'sam-tight':
        print(list(sam.starmap([(name,'sam3-box6') for name in batch.split(',')])))
    elif task == 'sam-wide':
        print(list(sam.starmap([(name,'sam3-box6') for name in ['dense-wide-full-gpu','dense-wide-discovery-gpu']])))
    elif task == 'transfer':
        clips = ['pittsburgh-finals-0600','southjersey-finals-0600','pittsburgh-finals-1200','southjersey-finals-1200']
        print(list(sam.starmap([(f'transfer-{clip}-gpu','sam3') for clip in clips])))
    elif task == 'inspect':
        destination=ROOT/'build/ball-temporal-20261026/sam-source-inspection.json'
        import json
        destination.write_text(json.dumps(inspect_source.remote(),indent=2)+'\n')
    else: raise ValueError(task)


@app.function(image=extend(sam_image), cpu=2, memory=8192, timeout=600)
def inspect_source():
    paths=['sam3/model_builder.py','sam3/model/sam3_tracking_predictor.py','sam3/model/sam3_tracker_base.py','sam3/model/sam3_base.py']
    return {p:(Path('/opt/sam3')/p).read_text()for p in paths if (Path('/opt/sam3')/p).exists()}
