"""Run the frozen Terra experiment against existing NBHL frames on Modal."""
from pathlib import Path
import modal

HERE = Path(__file__).parent
ROOT = HERE.parents[1] if modal.is_local() else Path('/work')
app = modal.App('sport-scans-sog-terra-20261027')
volume = modal.Volume.from_name('sport-scans-experiments')
image = (modal.Image.debian_slim(python_version='3.12').pip_install('pillow==11.3.0')
    .add_local_dir(HERE, '/work/experiment', ignore=['results', '__pycache__'])
    .add_local_dir(ROOT/'experiments/sog-20261027', '/work/baseline', ignore=['__pycache__'])
    .add_local_dir(ROOT/'hockey_tracking', '/work/hockey_tracking', ignore=['__pycache__'])
    .add_local_file(ROOT/'scripts/propose_shot_windows.py', '/work/propose_shot_windows.py')
    .add_local_file(ROOT/'scripts/detect_shots.py', '/work/detect_shots.py')
    .add_local_file(ROOT/'experiments/openai-20261024/run_nbhl_openai_pilot.py', '/work/client.py'))

@app.function(image=image, cpu=4, memory=8192, timeout=7200, volumes={'/data': volume}, max_containers=1)
def execute():
    import sys
    sys.path[:0] = ['/work/experiment', '/work/baseline', '/work']
    import worker_terra
    volume.reload()
    return worker_terra.run(volume)

@app.local_entrypoint()
def main(env_file: str = ''):
    import sys
    sys.path.insert(0, str(ROOT/'experiments/openai-20261024'))
    from run_nbhl_openai_pilot import read_key
    secret = modal.Secret.from_dict({'OPENAI_API_KEY': read_key(Path(env_file))})
    print(execute.with_options(secrets=[secret]).remote())
