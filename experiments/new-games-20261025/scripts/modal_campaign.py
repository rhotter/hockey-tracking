"""Bounded cloud experiments. Run with modal run -e main scripts/modal_campaign.py."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("sport-scans-experiments")
volume = modal.Volume.from_name("sport-scans-experiments", create_if_missing=True)
base_image = (modal.Image.from_registry("python:3.12-slim-bookworm")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("ultralytics==8.4.181", "numpy==1.26.4", "opencv-python==4.11.0.86",
                 "scipy", "lap==0.5.13", "cython-bbox", "huggingface_hub", "hydra-core",
                 "iopath", "einops", "tqdm", "pillow", "matplotlib", "pandas", "pycocotools",
                 "loguru", "filterpy", "timm", "yacs", "gdown", "tensorboard", "tabulate")
    .run_commands("git clone https://github.com/holma91/selective-mask-propagation.git /opt/smp && cd /opt/smp && git checkout 6472b52e79c47a7793cd4c6cabd64e812a8e5852",
                  "cd /opt/smp/vendor/sam2 && SAM2_BUILD_CUDA=0 pip install --no-build-isolation --no-deps -e .")
    .env({"PYTHONPATH": "/opt/smp:/work", "HF_HOME": "/data/cache/huggingface", "YOLO_CONFIG_DIR": "/data/cache/ultralytics"}))
