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
