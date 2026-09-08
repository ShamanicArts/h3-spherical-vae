"""Render known spherical references and pixel-only polar interpolation controls."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from spherical_projection import erp_directions,perspective_directions,rotate_directions,sample_erp

def texture(d):
    # A continuous spherical signal: no ERP edge or pole singularity.
    base=(d+1)/2
    features=.5+.5*np.sin(9*d[...,0]+4*d[...,2])*np.cos(7*d[...,1]-3*d[...,2])
    return np.clip(.7*base+.3*features[...,None],0,1)

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    erp=texture(erp_directions(672,1536));Image.fromarray(np.round(erp*255).astype('uint8')).save(a.output/'analytic-erp.png')
    view=perspective_directions(512,512,75);report={}
    for name,sign in [('north',1),('south',-1)]:
        rays=rotate_directions(view,np.array([[1,0,0],[0,0,sign],[0,-sign,0]]));reference=texture(rays);sampled=sample_erp(erp,rays)
        for suffix,pixels in [('direct',reference),('from-erp',sampled)]:Image.fromarray(np.round(pixels*255).astype('uint8')).save(a.output/f'{name}-{suffix}.png')
        error=sampled-reference;report[name]={'float_pixel_rmse':float(np.sqrt(np.mean(error**2))),'max_error':float(np.abs(error).max()),'h_fov':75,'width':512,'height':512}
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
