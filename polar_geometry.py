"""CPU ERP / rectilinear projection and localized repair composition.

Arrays are HWC (or HW), and values are kept in their input numerical range.
Returned images are float32, without clipping or quantization. ERP is a full
sphere regardless of its stored aspect ratio. Longitude zero is ERP center;
yaw 180 faces the horizontal wrap, matching the contact-sheet viewer.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    hfov_deg: float = 90.0
    yaw_deg: float = 180.0
    pitch_deg: float = 0.0

    def __post_init__(self):
        if self.width < 2 or self.height < 2:
            raise ValueError('View dimensions must each be at least two pixels')
        if not np.isfinite([self.hfov_deg, self.yaw_deg, self.pitch_deg]).all():
            raise ValueError('Camera angles must be finite')
        if not 0 < self.hfov_deg < 170 or not -90 <= self.pitch_deg <= 90:
            raise ValueError('Require 0 < horizontal FOV < 170 and |pitch| <= 90')

    @property
    def scale(self):
        tx = np.tan(np.deg2rad(self.hfov_deg) / 2)
        return tx, tx * self.height / self.width

    @property
    def basis(self):
        yaw, pitch = np.deg2rad([self.yaw_deg, self.pitch_deg])
        f = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
        r = np.array([np.cos(yaw), 0., -np.sin(yaw)])
        u = np.array([-np.sin(yaw)*np.sin(pitch), np.cos(pitch), -np.cos(yaw)*np.sin(pitch)])
        return f, r, u


def _image(value):
    a = np.asarray(value, dtype=np.float32)
    if a.ndim not in (2, 3) or min(a.shape[:2]) < 2 or not np.isfinite(a).all():
        raise ValueError('Expected a finite HW or HWC image, at least 2 by 2')
    return a


def _bilinear(a, x, y, wrap_x=False):
    """Coordinates are indexed at pixel centers; ERP wraps x, clamps poles."""
    h, w = a.shape[:2]
    x = np.mod(x, w) if wrap_x else np.clip(x, 0, w-1)
    y = np.clip(y, 0, h-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1 = (x0+1) % w if wrap_x else np.minimum(x0+1, w-1)
    y1 = np.minimum(y0+1, h-1)
    dx, dy = x-x0, y-y0
    if a.ndim == 3:
        dx, dy = dx[..., None], dy[..., None]
    return ((a[y0,x0]*(1-dx)+a[y0,x1]*dx)*(1-dy)
            +(a[y1,x0]*(1-dx)+a[y1,x1]*dx)*dy).astype(np.float32)


def view_rays(camera):
    """Unit world-space ray at every perspective pixel center."""
    tx, ty = camera.scale
    x, y = np.meshgrid((2*(np.arange(camera.width)+.5)/camera.width-1)*tx,
                       (1-2*(np.arange(camera.height)+.5)/camera.height)*ty)
    f, r, u = camera.basis
    d = f + x[...,None]*r + y[...,None]*u
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def extract_view(erp, camera):
    erp = _image(erp)
    h, w = erp.shape[:2]
    d = view_rays(camera)
    x = (np.arctan2(d[...,0], d[...,2])/(2*np.pi)+.5)*w-.5
    y = (.5-np.arcsin(np.clip(d[...,1],-1,1))/np.pi)*h-.5
    return _bilinear(erp, x, y, wrap_x=True)


def feather_mask(camera, core_fraction=.55, outer_fraction=.85):
    """Smooth rectangular view mask with zero support near all view edges.

    Fractions refer to half-width / half-height in perspective coordinates.
    Keep the repair central; do not blend the frustum's outer sampling boundary.
    """
    if not 0 <= core_fraction < outer_fraction < 1:
        raise ValueError('Require 0 <= core_fraction < outer_fraction < 1')
    x, y = np.meshgrid(abs(2*(np.arange(camera.width)+.5)/camera.width-1),
                       abs(2*(np.arange(camera.height)+.5)/camera.height-1))
    def ramp(d):
        t = np.clip((outer_fraction-d)/(outer_fraction-core_fraction),0,1)
        return t*t*(3-2*t)
    return (ramp(x)*ramp(y)).astype(np.float32)


def _inverse_grid(erp_shape, camera):
    h, w = erp_shape[:2]
    if h < 2 or w < 2:
        raise ValueError('ERP dimensions must be at least two')
    lon, lat = np.meshgrid(((np.arange(w)+.5)/w-.5)*2*np.pi,
                           (.5-(np.arange(h)+.5)/h)*np.pi)
    d = np.stack((np.sin(lon)*np.cos(lat), np.sin(lat), np.cos(lon)*np.cos(lat)),axis=-1)
    f, r, u = camera.basis
    z = d@f
    safe_z = np.where(z > 0,z,1.)
    tx, ty = camera.scale
    x = ((d@r)/safe_z/tx+1)*camera.width/2-.5
    y = (1-(d@u)/safe_z/ty)*camera.height/2-.5
    # Do not extrapolate beyond the first/last perspective pixel center.
    support = (z>0)&(x>=0)&(x<=camera.width-1)&(y>=0)&(y<=camera.height-1)
    return x, y, support


def project_view_to_erp(view, erp_shape, camera):
    """Backproject view; return (image, boolean geometric support).

    Unsupported output is zero, never edge-clamped content outside the frustum.
    """
    view = _image(view)
    if view.shape[:2] != (camera.height,camera.width):
        raise ValueError('View shape differs from camera')
    x, y, support = _inverse_grid(erp_shape,camera)
    result = _bilinear(view,x,y)
    result[~support] = 0
    return result, support


def composite_repair(original_erp, repaired_view, camera, mask=None,
                     mode='residual', reference_view=None, strength=1.):
    """Return (repaired ERP, ERP alpha), retaining outside pixels exactly.

    residual: original + alpha * backproject(repaired_view - reference_view).
    replace:  original*(1-alpha) + alpha * backproject(repaired_view).

    For a no-op control use extract_view(original,camera) as repaired_view.
    Residual no-op is exact; replace exposes projection interpolation changes.
    reference_view must be the *actual* image supplied to a repair model, after
    any input quantization, so that quantization is not mistaken for repair.
    """
    original = _image(original_erp)
    repaired = _image(repaired_view)
    if mode not in ('residual','replace') or not 0 <= strength <= 1:
        raise ValueError('Unknown composition mode or strength outside [0,1]')
    if repaired.shape[:2] != (camera.height,camera.width) or repaired.shape[2:] != original.shape[2:]:
        raise ValueError('Repaired view dimensions/channels differ')
    mask = feather_mask(camera) if mask is None else _image(mask)
    if mask.shape != (camera.height,camera.width) or np.any((mask<0)|(mask>1)):
        raise ValueError('Mask must be HW, finite, in [0,1] and match view')
    x, y, support = _inverse_grid(original.shape,camera)
    alpha = _bilinear(mask,x,y)*support*strength
    if mode == 'residual':
        reference = extract_view(original,camera) if reference_view is None else _image(reference_view)
        if reference.shape != repaired.shape:
            raise ValueError('Reference/repaired view shapes differ')
        delta = _bilinear(repaired-reference,x,y)
    else:
        delta = _bilinear(repaired,x,y)-original
    weight = alpha[...,None] if original.ndim == 3 else alpha
    result = original + weight*delta
    result[alpha==0] = original[alpha==0]
    return result.astype(np.float32), alpha.astype(np.float32)
