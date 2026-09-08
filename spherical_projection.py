"""Pixel-space spherical coordinates for controlled polar experiments.

No latent-channel transformation is implied. Directions use +Y up, +Z forward;
ERP longitude runs -pi..pi and latitude +pi/2..-pi/2 at pixel centres.
"""
import numpy as np

def erp_directions(height,width):
    if type(height) is not int or type(width) is not int or min(height,width)<2:raise ValueError('Invalid ERP size')
    lon=((np.arange(width)+.5)/width*2-1)*np.pi
    lat=(.5-(np.arange(height)+.5)/height)*np.pi
    lon,lat=np.meshgrid(lon,lat)
    return np.stack((np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)),-1)

def perspective_directions(height,width,horizontal_fov=75):
    if min(height,width)<2 or not 0<horizontal_fov<180:raise ValueError('Invalid perspective canvas/FOV')
    scale=np.tan(np.deg2rad(horizontal_fov)/2)
    x=((np.arange(width)+.5)/width*2-1)*scale
    y=(1-(np.arange(height)+.5)/height*2)*scale*height/width
    x,y=np.meshgrid(x,y);r=np.stack((x,y,np.ones_like(x)),-1)
    return r/np.linalg.norm(r,axis=-1,keepdims=True)

def rotate_directions(directions,matrix):
    matrix=np.asarray(matrix,dtype=np.float64)
    if matrix.shape!=(3,3) or not np.allclose(matrix.T@matrix,np.eye(3),atol=1e-9) or not np.isclose(np.linalg.det(matrix),1):raise ValueError('Expected a proper 3D rotation')
    return np.asarray(directions)@matrix.T

def sample_erp(image,directions):
    """Bilinear pixel sampling: periodic longitude and clamped cell-centred poles.

    Clamping is an explicit interpolation convention; use a directly evaluated
    analytic reference to quantify its error at the exact poles.
    """
    image=np.asarray(image); d=np.asarray(directions,dtype=np.float64)
    if image.ndim!=3 or min(image.shape[:2])<2 or d.shape[-1]!=3 or not np.isfinite(d).all():raise ValueError('Expected HWC pixels and finite directions')
    length=np.linalg.norm(d,axis=-1,keepdims=True)
    if np.any(length==0):raise ValueError('Zero direction')
    d=d/length;h,w=image.shape[:2]
    x=(np.arctan2(d[...,0],d[...,2])/(2*np.pi)+.5)*w-.5
    y=(.5-np.arcsin(np.clip(d[...,1],-1,1))/np.pi)*h-.5;y=np.clip(y,0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);fx=(x-x0)[...,None];fy=(y-y0)[...,None]
    a=image[y0,x0%w].astype(np.float64);b=image[y0,(x0+1)%w].astype(np.float64)
    c=image[np.minimum(y0+1,h-1),x0%w].astype(np.float64);e=image[np.minimum(y0+1,h-1),(x0+1)%w].astype(np.float64)
    return (a+(b-a)*fx)*(1-fy)+(c+(e-c)*fx)*fy

def solid_angle_weights(height,width):
    edges=np.linspace(np.pi/2,-np.pi/2,height+1)
    return np.broadcast_to((np.sin(edges[:-1])-np.sin(edges[1:]))[:,None]*(2*np.pi/width),(height,width)).copy()
