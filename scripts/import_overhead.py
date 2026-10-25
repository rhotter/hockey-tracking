"""Verify a Modal export and preserve all earlier pinned media in the v5 bundle."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
TAG='overhead-20261025'
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb')as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--export-report',type=Path,required=True);p.add_argument('--previous-assets',type=Path,required=True);a=p.parse_args()
    report=json.loads(a.export_report.read_text());old=json.loads((ROOT/'site-assets.json').read_text())
    assert sha(a.archive)==report['archive_sha256']and a.archive.stat().st_size==report['bytes']
    assert sha(a.previous_assets)==old['sha256']
    additions={};publication={}
    with tarfile.open(a.archive,'r:gz')as tar:
        members=tar.getmembers();assert len({m.name for m in members})==len(members)
        manifest=json.load(tar.extractfile('manifest.json'));assert {m.name for m in members}==set(manifest)|{'manifest.json'}
        for m in members:
            if m.name=='manifest.json':continue
            assert m.isfile()and '..'not in Path(m.name).parts
            assert m.name.startswith(('data/'+TAG+'/','media/'+TAG+'/'))
            content=tar.extractfile(m).read();digest=hashlib.sha256(content).hexdigest();assert digest==manifest[m.name]
            publication[m.name]=digest
            if m.name.startswith('data/'):
                assert Path(m.name).suffix=='.json';dest=ROOT/'web'/m.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
            else:
                assert Path(m.name).suffix in ['.jpg','.mp4','.png','.csv'];assert m.name not in old['files'];additions[m.name]=content
    for name in ['player-review.json','selection.json','transition-review.json','floor-line-review-v1.json','floor-line-review-v2.json','calibration-v1.json']:
        assert sha(ROOT/'web/data'/TAG/name)==sha(ROOT/'experiments'/TAG/'data'/name)
    archive=ROOT/'build/replay-assets.tar.gz';assert archive.resolve()!=a.previous_assets.resolve();files=dict(old['files'])
    with tarfile.open(a.previous_assets,'r:gz')as source,tarfile.open(archive,'w:gz',compresslevel=3)as dest:
        seen=set()
        for m in source:
            assert m.isfile()and m.name in files and m.name not in seen
            content=source.extractfile(m).read();assert hashlib.sha256(content).hexdigest()==files[m.name]
            dest.addfile(m,io.BytesIO(content));seen.add(m.name)
        assert seen==set(files)
        for name,content in sorted(additions.items()):
            m=tarfile.TarInfo(name);m.size=len(content);m.mode=0o644;dest.addfile(m,io.BytesIO(content));files[name]=hashlib.sha256(content).hexdigest()
    manifest={'schema':1,'url':'https://github.com/rhotter/hockey-tracking/releases/download/site-assets-v5/replay-assets-v5.tar.gz','sha256':sha(archive),'files':files}
    (ROOT/'site-assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'web/data'/TAG/'export-manifest.json').write_text(json.dumps({'archive_sha256':report['archive_sha256'],'files':publication},indent=2)+'\n')
    print(json.dumps({'assets_added':len(additions),'assets_total':len(files),'bytes':archive.stat().st_size,'sha256':manifest['sha256']}))
if __name__=='__main__':main()
