"""Wrapped-context decoding for the raw ComfyUI MiniMaxH3VideoVAE interface."""
from spherical_context import ContextPlan


def decode_circular(vae, video_latent, *, context_pixels=128, spatial_scale=16,
                    **decode_kwargs):
    """Decode BCTHW video latents with horizontal context, then return original canvas.

    Pass only the video stream, already in the native VAE's latent convention.
    Caller controls device/dtype and may supply the native `output_buffer` keyword.
    No latent normalization, denoising, RGB blending or vertical padding is applied.
    An output_buffer must have the *extended* shape from decode_output_shape().
    This adapter is for the raw native VAE, not ComfyUI's generic VAE wrapper.
    """
    if video_latent.ndim != 5:
        raise ValueError('Expected BCTHW video latent')
    if type(spatial_scale) is not int or spatial_scale < 1:
        raise ValueError('spatial_scale must be a positive integer')
    if type(context_pixels) is not int or context_pixels < 0 or context_pixels % spatial_scale:
        raise ValueError('Context must be a nonnegative whole number of latent columns')
    height,width=video_latent.shape[-2:]
    plan=ContextPlan(height,width,context_pixels//spatial_scale,0,'none')
    extended=plan.apply_torch(video_latent)
    expected=tuple(vae.decode_output_shape(extended.shape))
    if expected[-2:] != (height*spatial_scale,(width+2*plan.horizontal)*spatial_scale):
        raise ValueError('Decoder spatial scale does not match the declared scale')
    pixels=vae.decode(extended,**decode_kwargs)
    if tuple(pixels.shape) != expected:
        raise ValueError('Decoded output differs from the native shape contract')
    return plan.crop(pixels,spatial_scale)
