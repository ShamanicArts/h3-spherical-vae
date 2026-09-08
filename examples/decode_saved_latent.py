"""Decode a caller-supplied raw H3 video latent with an extended CPU output buffer.

Model loading is intentionally owned by the surrounding ComfyUI environment.
No file loading, cloud submission, or executable work occurs on import.
"""
import torch
from circular_decode import decode_circular


def decode_with_cpu_buffer(vae, video_latent, context_pixels=128):
    """Return BCTHW pixels using the original harness's CPU buffer convention.

    video_latent must already be on the raw VAE's device/dtype and use its native
    latent normalization. The returned spatial crop is generally noncontiguous
    and retains the backing buffer; downstream consumers may require a copy.
    """
    if (type(context_pixels) is not int or context_pixels < 0
            or context_pixels % 16 or context_pixels // 16 >= video_latent.shape[-1]):
        raise ValueError('Context must be whole latent columns, smaller than width')
    shape = list(video_latent.shape)
    shape[-1] += 2 * (context_pixels // 16)
    buffer = torch.empty(vae.decode_output_shape(shape), dtype=torch.float32, device='cpu')
    with torch.inference_mode():
        return decode_circular(vae, video_latent, context_pixels=context_pixels,
                               output_buffer=buffer)
