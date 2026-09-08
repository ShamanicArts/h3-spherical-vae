"""Final-step H3 chart prediction, with one accepted latent column per edge.

T2V only. Comfy packs video and audio into [B, 1, N]. The model continues to
use its original positional chart. This module does not alter model weights.
"""
import math
import numbers
import torch


def validate_shapes(latent_shapes):
    shapes=tuple(tuple(s) for s in latent_shapes)
    if len(shapes)!=2 or len(shapes[0])!=5 or len(shapes[1])!=4:
        raise ValueError('Expected video BCTHW and audio BC2T')
    if any(isinstance(n,bool) or not isinstance(n,numbers.Integral) or n<=0 for s in shapes for n in s):
        raise ValueError('Dimensions must be positive integers')
    v,a=shapes
    if v[0]!=a[0] or v[1]!=24 or a[1:3]!=(32,2) or v[-2]%2 or v[-1]%4:
        raise ValueError('H3 requires matching batches, 24/32 channels, even height, width divisible by four')
    return shapes


def _check(x,shapes):
    size=sum(math.prod(s[1:]) for s in shapes)
    if not isinstance(x,torch.Tensor) or getattr(x,'is_nested',False) or tuple(x.shape)!=(shapes[0][0],1,size) or not x.is_floating_point():
        raise ValueError('Expected floating packed H3 video/audio tensor')


def roll_video(x,columns,shapes):
    """Video-only permutation. Audio values and ordering remain exact."""
    shapes=validate_shapes(shapes); _check(x,shapes)
    if type(columns) is not int or columns%2 or abs(columns)>=shapes[0][-1]:
        raise ValueError('Shift must align with H3 patches and be smaller than width')
    count=math.prod(shapes[0][1:])
    video=torch.roll(x[...,:count].reshape(shapes[0]),columns,-1)
    return torch.cat((video.reshape(shapes[0][0],1,count),x[...,count:]),-1)


def accept_edge_columns(canonical,alternate,shapes):
    """Preserve the tested float32 correction arithmetic, including rounding."""
    shapes=validate_shapes(shapes); _check(canonical,shapes); _check(alternate,shapes)
    if canonical.dtype!=alternate.dtype or canonical.device!=alternate.device:
        raise ValueError('Predictions must share dtype and device')
    count=math.prod(shapes[0][1:]); a=canonical[...,:count].reshape(shapes[0]); b=alternate[...,:count].reshape(shapes[0])
    out=a.clone(); cols=torch.tensor([a.shape[-1]-1,0],device=a.device)
    av,bv=a.index_select(-1,cols).float(),b.index_select(-1,cols).float()
    out[...,cols]=(av+(bv-av)).to(a.dtype)
    return torch.cat((out.reshape(shapes[0][0],1,count),canonical[...,count:]),-1)


def build_final_shift_euler(latent_shapes,*,enabled=True,receipt=None,final_observer=None):
    """Comfy KSAMPLER callable: N ordinary evaluations, one extra at the end.

    final_observer receives detached-by-caller packed state, canonical clean,
    corrected clean, canonical final output and treated final output. It is an
    audit hook, called before returning, and must not mutate its inputs.
    Both outputs are from the same real forward call, avoiding reload drift.
    Only full, strictly descending flow schedules ending at zero are accepted.
    """
    shapes=validate_shapes(latent_shapes)
    if type(enabled) is not bool: raise ValueError('enabled must be boolean')
    @torch.no_grad()
    def sample(model,x,sigmas,extra_args=None,callback=None,disable=None):
        _check(x,shapes); args=extra_args or {}
        if not isinstance(sigmas,torch.Tensor) or sigmas.ndim!=1 or len(sigmas)<2 or not sigmas.is_floating_point():
            raise ValueError('Expected floating sigma schedule')
        if not torch.isfinite(sigmas).all() or sigmas[0]>1 or sigmas[-1]!=0 or torch.any(sigmas[:-1]<=sigmas[1:]):
            raise ValueError('Flow sigmas must strictly descend from <=1 to zero')
        if getattr(model,'denoise_mask',None) is not None or args.get('denoise_mask') is not None:
            raise ValueError('Spatial masks are not supported')
        for initial in (getattr(model,'latent_image',None),args.get('latent_image')):
            if initial is not None and (not isinstance(initial,torch.Tensor) or torch.count_nonzero(initial).item()):
                raise ValueError('Only zero initial clean latents are supported')
        sigma_in=x.new_ones([x.shape[0]])
        def predict(value,i):
            out=model(value,sigmas[i]*sigma_in,**args); _check(out,shapes)
            if out.dtype!=x.dtype or out.device!=x.device or not torch.isfinite(out).all(): raise ValueError('Invalid model prediction')
            return out
        for i in range(len(sigmas)-1):
            canonical=predict(x,i); clean=canonical; nfe=1
            final=i==len(sigmas)-2
            if enabled and final:
                shift=shapes[0][-1]//2
                alternate=roll_video(predict(roll_video(x,shift,shapes),i),-shift,shapes)
                clean=accept_edge_columns(canonical,alternate,shapes); nfe=2
            if callback is not None: callback({'x':x,'i':i,'sigma':sigmas[i],'sigma_hat':sigmas[i],'denoised':clean})
            result=x+(x-clean)/sigmas[i]*(sigmas[i+1]-sigmas[i])
            if final and final_observer is not None:
                control=x+(x-canonical)/sigmas[i]*(sigmas[i+1]-sigmas[i])
                final_observer(x,canonical,clean,control,result)
            if receipt is not None: receipt.append({'step':i+1,'sigma':float(sigmas[i]),'sigma_next':float(sigmas[i+1]),'model_evaluations':nfe,'accepted_edge_columns':1 if enabled and final else 0})
            x=result
        return x
    return sample
