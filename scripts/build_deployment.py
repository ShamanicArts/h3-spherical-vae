"""Build an isolated fal bundle. This command does not launch GPU compute."""
import argparse, hashlib, json, shutil, subprocess, sys, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
COMFY='12d5279438bfefc058a269eae805ceab6047777f'
SHA='2fb9bb75a0c2c377f408fea2f06c3b68ccf33505ed3c1332d830e4d5cb44bc82'

def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('destination',type=Path);p.add_argument('--comfy-archive',type=Path);args=p.parse_args()
    dest=args.destination.resolve();dest.mkdir(parents=True,exist_ok=False)
    # Config is explicit deployment input; never discover local credentials.
    cfg=json.loads(args.config.read_text())
    if not 1<=cfg['worker_guard_seconds']<=3600:raise ValueError('Worker guard must be <= one hour')
    if not cfg['run_id'] or '/' in cfg['run_id']:raise ValueError('Invalid run ID')
    subprocess.run(['uv','build','--wheel',str(ROOT),'--out-dir',str(dest/'wheel')],check=True)
    wheel,=list((dest/'wheel').glob('*.whl'));shutil.copyfile(wheel,dest/'runtime.whl');shutil.rmtree(dest/'wheel')
    for name in ('fal_app.py','models.json'):shutil.copyfile(ROOT/'deployment'/name,dest/name)
    shutil.copyfile(args.config,dest/'config.json')
    target=dest/'comfy-source.tar.gz'
    if args.comfy_archive:shutil.copyfile(args.comfy_archive,target)
    else:urllib.request.urlretrieve(f'https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/{COMFY}',target)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=SHA:raise ValueError('Pinned Comfy archive hash mismatch')
    receipt={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}
    (dest/'bundle-sha256.json').write_text(json.dumps(receipt,indent=2)+'\n');print(dest)
if __name__=='__main__':main()
