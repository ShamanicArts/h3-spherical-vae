import unittest
import numpy as np
from spherical_projection import erp_directions,perspective_directions,rotate_directions,sample_erp,solid_angle_weights
class ProjectionTests(unittest.TestCase):
    def test_identity_and_half_turn(self):
        rng=np.random.default_rng(360);image=rng.random((32,64,3));d=erp_directions(32,64)
        np.testing.assert_allclose(sample_erp(image,d),image,atol=1e-12)
        rotated=rotate_directions(d,np.diag([-1,1,-1]))
        np.testing.assert_allclose(sample_erp(image,rotated),np.roll(image,32,axis=1),atol=1e-12)
    def test_pole_views_against_analytic_sphere(self):
        d=erp_directions(256,512);pixels=(d+1)/2;view=perspective_directions(64,64)
        for sign in (-1,1):
            rotation=np.array([[1,0,0],[0,0,sign],[0,-sign,0]])
            pole=rotate_directions(view,rotation)
            self.assertLess(np.max(np.abs(sample_erp(pixels,pole)-(pole+1)/2)),.0001)
    def test_sphere_area(self):self.assertAlmostEqual(float(solid_angle_weights(128,256).sum()),4*np.pi)
    def test_reject_nonrotation(self):
        with self.assertRaises(ValueError):rotate_directions(erp_directions(2,4),np.diag([1,1,-1]))
if __name__=='__main__':unittest.main()
