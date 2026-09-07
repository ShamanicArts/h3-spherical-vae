import unittest
try:
    import torch
except ImportError:
    torch = None
from periodic_tiles import periodic_tiled_decode


class IdentityDecoder:
    vae_ratio = 1
    tile_size = 8

    def split_tiles(self, height):
        return ([0, 4], [8, 8], [4]) if height == 12 else ([0], [height], [])

    def _decode_pixels(self, z):
        return z.clone()

    def blend(self, a, b, extent, dim):
        result = b.clone()
        w = torch.arange(extent, dtype=b.dtype) / extent
        result[..., :extent, :] = a[..., -extent:, :] * (1-w[:, None]) + b[..., :extent, :] * w[:, None]
        return result


@unittest.skipIf(torch is None, 'Optional decoder adapter requires PyTorch')
class PeriodicTileTests(unittest.TestCase):
    def test_reconstruction_coverage_and_immutable_inputs(self):
        z = torch.randn(1, 3, 2, 12, 24)
        original = z.clone()
        for blend in ('linear', 'cosine'):
            for phase in (0, 2, 4, 22):
                actual = periodic_tiled_decode(IdentityDecoder(), z, blend=blend, phase_pixels=phase)
                torch.testing.assert_close(actual, z, rtol=1e-6, atol=1e-6)
                self.assertTrue(torch.equal(original, z))

    def test_reconstructs_nearest_expansion_and_dimensions(self):
        class Expanded(IdentityDecoder):
            vae_ratio = 2
            def _decode_pixels(self, z):
                return z.repeat_interleave(2,-2).repeat_interleave(2,-1)
        z = torch.randn(1, 3, 2, 6, 12)
        actual = periodic_tiled_decode(Expanded(), z, phase_pixels=2)
        torch.testing.assert_close(actual,z.repeat_interleave(2,-2).repeat_interleave(2,-1),rtol=1e-6,atol=1e-6)

    def test_circular_neighbour_operator_stays_correct_at_wrap(self):
        # Zero-weight endpoints of a triangular window are not assumed. Compare
        # phase translations of the entire tiled operator, including tile edges.
        class Local(IdentityDecoder):
            def _decode_pixels(self,z):
                return z + .2*torch.roll(z,1,-1)
        z=torch.randn(1,3,2,12,24)
        for blend in ('linear','cosine'):
            original=periodic_tiled_decode(Local(),z,blend=blend)
            shifted=periodic_tiled_decode(Local(),torch.roll(z,2,-1),blend=blend,phase_pixels=2)
            torch.testing.assert_close(shifted,torch.roll(original,2,-1),rtol=1e-6,atol=1e-6)

    def test_rejects_unaligned_geometry(self):
        with self.assertRaises(ValueError):
            periodic_tiled_decode(IdentityDecoder(), torch.zeros(1,3,2,12,23))
        with self.assertRaises(ValueError):
            periodic_tiled_decode(IdentityDecoder(), torch.zeros(1,3,2,12,24), blend='invalid')


if __name__ == '__main__':
    unittest.main()
