"""Build a deployable service bundle; never starts or deploys compute."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
COMFY='12d5279438bfefc058a269eae805ceab6047777f'
ARCHIVE_SHA='2fb9bb75a0c2c377f408fea2f06c3b68ccf33505ed3c1332d830e4d5cb44bc82'


def main():
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);p.add_argument('--comfy-archive',type=Path);args=p.parse_args()
    dest=args.destination.resolve();dest.mkdir(parents=True,exist_ok=False)
    subprocess.run(['uv','build','--wheel',str(ROOT),'--out-dir',str(dest/'wheel')],check=True)
    wheel,=list((dest/'wheel').glob('*.whl'));shutil.copyfile(wheel,dest/'runtime.whl');shutil.rmtree(dest/'wheel')
    with zipfile.ZipFile(dest/'runtime.whl') as z:
        required={'h3_runtime.py','polar_pipeline.py','polar_kernel.py','polar_geometry.py','service_schema.py','service_worker.py','final_shift.py','circular_decode.py'}
        if not required.issubset(z.namelist()):raise ValueError('Runtime wheel is incomplete')
    shutil.copyfile(ROOT/'deployment/service_app.py',dest/'service_app.py')
    shutil.copyfile(ROOT/'deployment/models.json',dest/'models.json')
    shutil.copyfile(ROOT/'service_schema.py',dest/'service_schema.py')
    target=dest/'comfy-source.tar.gz'
    if args.comfy_archive:shutil.copyfile(args.comfy_archive,target)
    else:urllib.request.urlretrieve(f'https://codeload.github.com/Comfy-Org/ComfyUI/tar.gz/{COMFY}',target)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=ARCHIVE_SHA:raise ValueError('Comfy archive differs')
    receipt={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}
    (dest/'bundle-sha256.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(dest)


if __name__=='__main__':main()
