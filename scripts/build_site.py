"""Assemble the website using a checksum-pinned replay asset release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request


def build(root, archive=None):
    manifest = json.loads((root / 'site-assets.json').read_text())
    output = root / 'dist'
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(root / 'web', output)
    if archive is None:
        archive = root / 'build/replay-assets.tar.gz'
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(manifest['url'], archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Replay archive checksum mismatch')
    seen = set()
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            name = member.name
            if not member.isfile() or name not in manifest['files'] or name in seen:
                raise ValueError(f'Unexpected archive member: {name}')
            dest = output / name
            if not dest.resolve().is_relative_to(output.resolve()) or dest.exists():
                raise ValueError(f'Invalid or duplicate asset path: {name}')
            content = bundle.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != manifest['files'][name]:
                raise ValueError(f'Asset checksum mismatch: {name}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            seen.add(name)
    if seen != set(manifest['files']):
        raise ValueError('Replay archive is missing required assets')
    (output / '.nojekyll').touch()
    print(f'Built website with {len(seen)} verified replay assets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    build(args.root.resolve(), args.archive)
