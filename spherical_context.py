"""Coordinate-only ERP context plans. No model, file or cloud side effects."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContextPlan:
    height: int
    width: int
    horizontal: int
    vertical: int
    vertical_mode: str

    def __post_init__(self):
        for value in (self.height, self.width, self.horizontal, self.vertical):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError('Grid sizes and margins must be integers')
        if self.height < 2 or self.width < 2:
            raise ValueError('Grid dimensions must be at least two')
        if not 0 <= self.horizontal < self.width or not 0 <= self.vertical < self.height:
            raise ValueError('Context margins must be smaller than the original grid')
        if self.vertical_mode not in ('none', 'reflect', 'sphere'):
            raise ValueError('Unknown vertical context treatment')
        if (self.vertical_mode == 'none') != (self.vertical == 0):
            raise ValueError('Use none exactly when there is no vertical margin')
        if self.vertical_mode == 'sphere' and self.width % 2:
            raise ValueError('Exact half-turn indexing requires even longitude width')

    def indices(self):
        rows = np.arange(-self.vertical, self.height+self.vertical)
        outside = (rows < 0) | (rows >= self.height)
        # Cell-centred reflection duplicates the boundary cell. This is different
        # from PyTorch reflect padding, which assumes a different boundary rule.
        source_rows = np.where(rows < 0, -rows-1,
                               np.where(rows >= self.height, 2*self.height-rows-1, rows))
        shift = outside.astype(np.int64)*(self.width//2) if self.vertical_mode == 'sphere' else np.zeros_like(rows)
        cols = (np.arange(-self.horizontal,self.width+self.horizontal)[None,:]+shift[:,None]) % self.width
        return source_rows, cols

    def crop(self, decoded, spatial_scale=1):
        if isinstance(spatial_scale, bool) or not isinstance(spatial_scale,int) or spatial_scale < 1:
            raise ValueError('Spatial scale must be a positive integer')
        expected=((self.height+2*self.vertical)*spatial_scale,
                  (self.width+2*self.horizontal)*spatial_scale)
        if tuple(decoded.shape[-2:]) != expected:
            raise ValueError(f'Expected decoded spatial shape {expected}')
        top,left=self.vertical*spatial_scale,self.horizontal*spatial_scale
        return decoded[...,top:top+self.height*spatial_scale,left:left+self.width*spatial_scale]

    def apply_numpy(self, grid):
        self._check_shape(grid)
        rows,cols=self.indices()
        return grid[...,rows[:,None],cols].copy()

    def apply_torch(self, grid):
        import torch
        self._check_shape(grid)
        rows,cols=self.indices()
        return grid[...,torch.as_tensor(rows,device=grid.device)[:,None],
                    torch.as_tensor(cols,device=grid.device)].clone()

    def _check_shape(self, grid):
        if grid.ndim < 2 or tuple(grid.shape[-2:]) != (self.height,self.width):
            raise ValueError('Input must end with the planned latitude/longitude grid')
