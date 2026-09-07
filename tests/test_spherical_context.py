import unittest
import numpy as np
from spherical_context import ContextPlan


class ContextTests(unittest.TestCase):
    def test_native_shapes_crop_and_immutable_source(self):
        a=np.arange(1*24*37*42*96,dtype=np.int32).reshape(1,24,37,42,96)
        before=a.copy()
        for mode,rows in [('none',0),('reflect',4),('sphere',4)]:
            plan=ContextPlan(42,96,8,rows,mode)
            padded=plan.apply_numpy(a)
            self.assertEqual(padded.shape,(1,24,37,42+rows*2,112))
            np.testing.assert_array_equal(plan.crop(padded),a)
            padded[...] = -1
            np.testing.assert_array_equal(a,before)

    def test_horizontal_matches_existing_concatenation(self):
        a=np.arange(42*96).reshape(42,96)
        padded=ContextPlan(42,96,8,0,'none').apply_numpy(a)
        np.testing.assert_array_equal(padded,np.concatenate([a[:,-8:],a,a[:,:8]],axis=-1))

    def test_poles_and_corners(self):
        a=np.arange(6*8).reshape(6,8)
        p=ContextPlan(6,8,2,2,'sphere').apply_numpy(a)
        np.testing.assert_array_equal(p[1,2:-2],np.roll(a[0],4))
        np.testing.assert_array_equal(p[-2,2:-2],np.roll(a[-1],4))
        self.assertEqual(p[0,0],a[1,2])
        self.assertEqual(p[-1,-1],a[-2,5])

    def test_reflection_control_uses_same_rows_without_half_turn(self):
        a=np.arange(6*8).reshape(6,8)
        p=ContextPlan(6,8,2,2,'reflect').apply_numpy(a)
        np.testing.assert_array_equal(p[1,2:-2],a[0])
        np.testing.assert_array_equal(p[-2,2:-2],a[-1])

    def test_analytic_sphere_agrees_across_both_poles(self):
        h,w,margin=42,96,4
        def xyz(rows,cols):
            lat=np.pi/2-(rows+.5)*np.pi/h
            lon=(cols+.5)*2*np.pi/w-np.pi
            lat,lon=np.broadcast_arrays(lat[:,None],lon[None,:])
            return np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)])
        sphere=xyz(np.arange(h),np.arange(w))
        actual=ContextPlan(h,w,8,margin,'sphere').apply_numpy(sphere)
        expected=xyz(np.arange(-margin,h+margin),np.arange(-8,w+8))
        np.testing.assert_allclose(actual,expected,atol=2e-15)

    def test_decoded_crop_keeps_native_dimensions(self):
        plan=ContextPlan(42,96,8,4,'sphere')
        image=np.zeros((1,3,124,800,1792),dtype=np.uint8)
        self.assertEqual(plan.crop(image,16).shape,(1,3,124,672,1536))

    def test_invalid_geometry_is_rejected(self):
        for args in [(42,95,8,4,'sphere'),(42,96,8,0,'sphere'),
                     (42,96,8,4,'none'),(42,96,96,4,'reflect')]:
            with self.assertRaises(ValueError): ContextPlan(*args)


if __name__=='__main__': unittest.main()
