"""Periodic horizontal tile assembly for a compatible tiled video decoder.

The adapter expects tile_size, vae_ratio, split_tiles, _decode_pixels and blend
on its decoder. It preserves the decoder's vertical schedule and temporal API.
It is experimental: valid coverage does not imply improved learned decoding.
"""
import math


def periodic_tiled_decode(decoder, z, *, blend='cosine', phase_pixels=0):
    import torch

    ratio = decoder.vae_ratio
    tile = decoder.tile_size
    height, width = z.shape[-2] * ratio, z.shape[-1] * ratio
    stride = tile // 2
    if blend not in ('linear', 'cosine'):
        raise ValueError('Unknown horizontal blend')
    if tile % (2 * ratio) or width % stride or width < tile or phase_pixels % ratio:
        raise ValueError('Periodic tiles require aligned size, half-stride, width and phase')
    y_starts, y_lengths, y_overlaps = decoder.split_tiles(height)
    x_starts = [(x + phase_pixels) % width for x in range(0, width, stride)]
    sample = (torch.arange(tile, device=z.device, dtype=torch.float32) + .5) / tile
    weights = (1 - torch.abs(2 * sample - 1)) if blend == 'linear' else torch.sin(math.pi * sample).square()
    denominator = torch.zeros(width, device=z.device, dtype=torch.float32)
    pixel_indices = [(torch.arange(tile, device=z.device) + x) % width for x in x_starts]
    for indices in pixel_indices:
        denominator.index_add_(0, indices, weights)
    if not torch.all(denominator > 0):
        raise RuntimeError('Uncovered output columns')

    canvas = None
    row_tails = []
    out_y = 0
    for i, (y, length) in enumerate(zip(y_starts, y_lengths)):
        new_tails = []
        row = None
        for j, x in enumerate(x_starts):
            columns = (torch.arange(tile // ratio, device=z.device) + x // ratio) % z.shape[-1]
            source = z[..., y // ratio:(y + length) // ratio, :].index_select(-1, columns)
            pixels = decoder._decode_pixels(source)
            if i < len(y_starts) - 1:
                new_tails.append(pixels[..., -y_overlaps[i]:, :].clone())
            if i:
                pixels = decoder.blend(row_tails[j], pixels, y_overlaps[i-1], dim=-2)
            if i < len(y_starts) - 1:
                pixels = pixels[..., :-y_overlaps[i], :]
            if row is None:
                row = torch.zeros(*pixels.shape[:-1], width, device=pixels.device, dtype=torch.float32)
            row.index_add_(-1, pixel_indices[j], pixels.float() * weights)
        row /= denominator
        if canvas is None:
            canvas = torch.empty(*row.shape[:-2], height, width, device=row.device, dtype=pixels.dtype)
        canvas[..., out_y:out_y + row.shape[-2], :].copy_(row)
        out_y += row.shape[-2]
        row_tails = new_tails
    if out_y != height:
        raise RuntimeError('Unexpected vertical output coverage')
    return canvas
