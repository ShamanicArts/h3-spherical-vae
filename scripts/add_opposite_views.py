"""Add matched opposite-longitude evidence. Supply the saved 100-step PNG directory."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--temple100-snapshots',type=Path,required=True); args=p.parse_args()
    folder=ROOT/'docs/report-card'; evidence=folder/'evidence.json'; d=json.loads(evidence.read_text()); checks=[]
    for case,frame in [('temple50',97),('forest50',97),('temple100',0),('temple100',123)]:
        sources={}
        for arm in ('control','final'):
            key=f'{case}-{arm}-circular128-f{frame:03d}'
            if case=='temple100':
                prefix='night-100r-control' if arm=='control' else 'night-100r-at100'
                source=args.temple100_snapshots/f'{prefix}-circular128-frame{frame:03d}.png'
                expected=d['images'][key+'-eye30']['source_png_sha256']
            else:
                entry=d['images'][key+'-erp']; source=folder/entry['file']; expected=entry['sha256']
            assert sha(source)==expected
            sources[arm]=np.asarray(Image.open(source).convert('RGB'))
            for fov,vfov in [(75,46.69),(20,11.34)]:
                dest=folder/f'assets/{key}-opposite{fov}.png'
                vf=f'v360=input=equirect:output=flat:yaw=0:pitch=0:h_fov={fov}:v_fov={vfov}:w=640:h=360'
                subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-vf',vf,'-frames:v','1',str(dest)],check=True)
                d['images'][key+f'-opposite{fov}']={'file':str(dest.relative_to(folder)),'sha256':sha(dest),'source_png_sha256':expected,'frame':frame,'arm':arm,'decoder':'circular128','projection':{'yaw_degrees':0,'pitch_degrees':0,'horizontal_fov_degrees':fov,'vertical_fov_degrees':vfov,'output_width':640,'output_height':360,'filter':vf}}
        a,b=sources['control'],sources['final']; w=a.shape[1]
        delta=np.abs(a[:,w//4:3*w//4].astype(np.int16)-b[:,w//4:3*w//4].astype(np.int16))
        checks.append({'case':case,'frame':frame,'region':'middle half of ERP width, all rows','max_rgb_difference':int(delta.max()),'mean_rgb_difference':float(delta.mean()),'changed_channels':int(np.count_nonzero(delta))})
    d['opposite_longitude_checks']=checks
    evidence.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
