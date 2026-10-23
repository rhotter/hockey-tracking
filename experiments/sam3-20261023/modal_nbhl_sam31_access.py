"""Check the separate SAM 3.1 checkpoint gate without exposing credentials."""
import json
from pathlib import Path
import modal
from modal_campaign import ROOT, base_image, volume

app = modal.App('sport-scans-nbhl-sam31-access')
REV = 'daa63191845a41281374e725f4c9e51c7a824460'
image = base_image.env({'HF_HOME': '/data/cache/huggingface', 'HF_HUB_DISABLE_XET': '1'})
image = image.add_local_file(ROOT/'scripts/modal_campaign.py', '/work/modal_campaign.py')

@app.function(image=image, cpu=2, memory=8192, timeout=1800,
              secrets=[modal.Secret.from_name('huggingface', required_keys=['HF_TOKEN'])],
              volumes={'/data': volume})
def check():
    import os, datetime
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    result = {'model': 'facebook/sam3.1', 'revision': REV,
              'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        path = hf_hub_download('facebook/sam3.1', 'sam3.1_multiplex.pt', revision=REV,
                               token=os.environ['HF_TOKEN'])
        result.update(access=True, bytes=Path(path).stat().st_size)
        volume.commit()
    except HfHubHTTPError as exc:
        result.update(access=False, error_type=type(exc).__name__,
                      status=exc.response.status_code if exc.response is not None else None)
    return result

@app.local_entrypoint()
def main():
    result = check.remote()
    (ROOT/'notes/ball-hockey/sam31-access-20261023.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
