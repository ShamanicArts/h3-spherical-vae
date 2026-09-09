import unittest
import numpy as np
from service_schema import SphericalRequest, REPAIR_PROMPT
from polar_pipeline import small_mask, composite


class ServiceContractTests(unittest.TestCase):
    def test_prompt_and_repair_instructions_are_independent(self):
        r=SphericalRequest(prompt='A temple under yellow skies')
        self.assertEqual([p[2] for p in r.poles()],[REPAIR_PROMPT]*2)
        self.assertNotIn('top_prompt',r.generation_config())
        r=SphericalRequest(prompt='Yellow sky',repair_top=False,bottom_prompt='')
        self.assertEqual(r.poles(),[('bottom',-90,'')])

    def test_no_silent_shape_or_parameter_changes(self):
        for bad in [{'frames':125},{'width':1536,'height':448},{'steps':101},{'repair_steps':0}, {'typo':True}]:
            with self.assertRaises(ValueError):SphericalRequest(prompt='ERP',**bad)
        with self.assertRaises(ValueError):SphericalRequest(video_url='file:///tmp/video')
        with self.assertRaises(ValueError):SphericalRequest(video_url='https://example.com/a.mp4',prompt='unused')

    def test_native_baseline_switches(self):
        r=SphericalRequest(prompt='ERP',circular_decode=False,final_shift=False,repair_top=False,repair_bottom=False)
        self.assertEqual(r.generation_config()['contexts'],[0])
        self.assertFalse(r.generation_config()['final_shift'])
        self.assertEqual(r.poles(),[])

    def test_mask_and_backprojection_identity_and_separation(self):
        a=small_mask();self.assertEqual(a.shape,(512,512))
        self.assertEqual(a[256,256],1);self.assertEqual(a[0,0],0)
        rng=np.random.default_rng(42)
        source=rng.integers(0,256,(2,64,128,3),dtype=np.uint8)
        reference=rng.integers(0,256,(2,512,512,3),dtype=np.uint8)
        for pitch in [-90,90]:
            self.assertTrue(np.array_equal(source,composite(source,reference,reference,pitch)))
        fixed=np.rint(reference+(255-reference)*a[None,:,:,None]).astype(np.uint8)
        top=composite(source,reference,fixed,90)
        bottom=composite(source,reference,fixed,-90)
        both=composite(top,reference,fixed,-90)
        self.assertTrue(np.array_equal(top[:,32:],source[:,32:]))
        self.assertTrue(np.array_equal(bottom[:,:32],source[:,:32]))
        self.assertTrue(np.array_equal(both[:,:32],top[:,:32]))
        self.assertTrue(np.array_equal(both[:,32:],bottom[:,32:]))
        self.assertFalse(np.array_equal(top,source))


if __name__=='__main__':unittest.main()
