"""Verify and prepare public assets for the Next.js app."""
from pathlib import Path
import shutil
from build_site import build

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'build/replay-assets.tar.gz'
build(ROOT, archive if archive.exists() else None)
public = ROOT / 'public'
if public.exists():
    shutil.rmtree(public)
shutil.copytree(ROOT / 'dist', public)
for page in public.glob('*.html'):
    page.unlink()
print('Next.js public assets prepared; pages are rendered by the App Router')
