import unittest
try:
    import torch
except ImportError:
    torch=None
from circular_decode import decode_circular

@unittest.skipIf(torch is None,'PyTorch supplied by model environment')
class DecodeTests(unittest.TestCase):
    def test_native_scale_crop_and_periodic_neighbors(self):
        class Decoder:
            def decode_output_shape(self,shape):return (*shape[:-2],shape[-2]*16,shape[-1]*16)
            def decode(self,x,**kw):
                self.seen=x.clone()
                return x.repeat_interleave(16,-2).repeat_interleave(16,-1)
        z=torch.arange(2*3*4*6*32).reshape(2,3,4,6,32).float();original=z.clone()
        for context in (0,128,384):
            vae=Decoder();out=decode_circular(vae,z,context_pixels=context)
            self.assertTrue(torch.equal(out,z.repeat_interleave(16,-2).repeat_interleave(16,-1)))
            n=context//16
            if n:
                self.assertTrue(torch.equal(vae.seen[...,:n],z[...,-n:]))
                self.assertTrue(torch.equal(vae.seen[...,-n:],z[...,:n]))
            self.assertTrue(torch.equal(z,original))
    def test_invalid_context_and_decoder_shape_rejected(self):
        z=torch.zeros(1,24,2,4,32)
        for value in (-1,17,True,512):
            with self.assertRaises(ValueError):decode_circular(None,z,context_pixels=value)
        class Bad:
            def decode_output_shape(self,s):return (1,3,5,3,7)
        with self.assertRaises(ValueError):decode_circular(Bad(),z)

    def test_cpu_output_buffer_example_preserves_native_contract(self):
        from examples.decode_saved_latent import decode_with_cpu_buffer

        class BufferedDecoder:
            def decode_output_shape(self, shape):
                return (shape[0], 3, shape[2], shape[3]*16, shape[4]*16)

            def decode(self, z, *, output_buffer):
                self.asserted_inference = torch.is_inference_mode_enabled()
                self.buffer = output_buffer
                self.seen = z.clone()
                output_buffer.copy_(z[:, :3].repeat_interleave(16, -2)
                                    .repeat_interleave(16, -1))
                return output_buffer

        z = torch.randn(1, 24, 2, 4, 32, dtype=torch.float16)
        for margin in (0, 128, 384):
            vae = BufferedDecoder()
            out = decode_with_cpu_buffer(vae, z, margin)
            self.assertTrue(vae.asserted_inference)
            self.assertEqual(vae.seen.dtype, z.dtype)
            self.assertEqual(vae.buffer.shape, (1, 3, 2, 64, 512 + 2*margin))
            self.assertEqual(out.shape, (1, 3, 2, 64, 512))
            self.assertEqual(out.dtype, torch.float32)
            self.assertEqual(out.device.type, 'cpu')
            self.assertTrue(torch.equal(out, z[:, :3].float()
                                       .repeat_interleave(16, -2)
                                       .repeat_interleave(16, -1)))

    def test_wrong_actual_decoded_shape_rejected(self):
        class Decoder:
            def decode_output_shape(self, s):
                return (s[0], 3, s[2], s[3]*16, s[4]*16)
            def decode(self, z):
                return torch.empty(1, 3, 2, 64, 12)
        with self.assertRaisesRegex(ValueError, 'Decoded output'):
            decode_circular(Decoder(), torch.zeros(1, 24, 2, 4, 32))
