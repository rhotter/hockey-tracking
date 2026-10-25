"""Verify the Modal export and append its media to the pinned website bundle."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TAG = 'new-games-20261025'


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--export-report', required=True, type=Path)
    parser.add_argument('--previous-assets', required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.export_report.read_text())
    assert sha(args.archive) == report['sha256']
    assert args.archive.stat().st_size == report['bytes']
    old = json.loads((ROOT / 'site-assets.json').read_text())
    assert sha(args.previous_assets) == old['sha256']
    additions = {}
    publication = {}
    with tarfile.open(args.archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        assert len({m.name for m in members}) == len(members)
        manifest_member = next(m for m in members if m.name == 'manifest.json')
        assert manifest_member.isfile()
        manifest = json.load(bundle.extractfile(manifest_member))
        assert {m.name for m in members} == set(manifest) | {'manifest.json'}
        for member in members:
            if member.name == 'manifest.json':
                continue
            assert member.isfile() and '..' not in Path(member.name).parts
            prefix, relative = member.name.split('/', 1)
            assert prefix in ['data', 'media'] and not relative.startswith('/')
            content = bundle.extractfile(member).read()
            digest = hashlib.sha256(content).hexdigest()
            assert digest == manifest[member.name]
            target = ('data/reliability/' if prefix == 'data' else 'media/') + TAG + '/' + relative
            assert (ROOT / 'web' / target).resolve().is_relative_to(ROOT / 'web')
            publication[target] = digest
            if Path(relative).suffix == '.json':
                dest = ROOT / 'web' / target
                assert not dest.exists() or sha(dest) == digest
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
            else:
                assert prefix == 'media' and Path(relative).suffix in ['.jpg', '.mp4']
                assert target not in old['files']
                additions[target] = content
    for name in ['protocol.json', 'reference.json', 'source-review.json']:
        assert sha(ROOT / 'web/data/reliability' / TAG / name) == sha(ROOT / 'experiments' / TAG / 'data' / name)
    archive = ROOT / 'build/replay-assets.tar.gz'
    assert archive.resolve() != args.previous_assets.resolve()
    files = dict(old['files'])
    with tarfile.open(args.previous_assets, 'r:gz') as source, tarfile.open(archive, 'w:gz', compresslevel=3) as dest:
        seen = set()
        for member in source:
            assert member.isfile() and member.name in files and member.name not in seen
            content = source.extractfile(member).read()
            assert hashlib.sha256(content).hexdigest() == files[member.name]
            dest.addfile(member, io.BytesIO(content))
            seen.add(member.name)
        assert seen == set(files)
        for name, content in sorted(additions.items()):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o644
            dest.addfile(member, io.BytesIO(content))
            files[name] = hashlib.sha256(content).hexdigest()
    manifest = {'schema': 1, 'url': 'https://github.com/rhotter/hockey-tracking/releases/download/site-assets-v4/replay-assets-v4.tar.gz', 'sha256': sha(archive), 'files': files}
    (ROOT / 'site-assets.json').write_text(json.dumps(manifest, indent=2) + '\n')
    output = {'archive_sha256': report['sha256'], 'files': publication}
    (ROOT / 'web/data/reliability' / TAG / 'export-manifest.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'assets_added': len(additions), 'assets_total': len(files), 'archive_bytes': archive.stat().st_size, 'sha256': manifest['sha256']}))


if __name__ == '__main__':
    main()
