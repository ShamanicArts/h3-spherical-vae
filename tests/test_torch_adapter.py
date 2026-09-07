import unittest
import numpy as np
from spherical_context import ContextPlan

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional model-environment adapter requires PyTorch')
class TorchAdapterTests(unittest.TestCase):
    def test_matches_numpy_and_preserves_gradient_paths(self):
        a=np.arange(2*3*6*8,dtype=np.float32).reshape(2,3,6,8)
        for mode,rows in [('none',0),('reflect',2),('sphere',2)]:
            plan=ContextPlan(6,8,2,rows,mode)
            grid=torch.tensor(a,requires_grad=True)
            result=plan.apply_torch(grid)
            np.testing.assert_array_equal(result.detach().numpy(),plan.apply_numpy(a))
            self.assertEqual(result.dtype,grid.dtype)
            plan.crop(result).sum().backward()
            self.assertTrue(torch.equal(grid.grad,torch.ones_like(grid)))
            np.testing.assert_array_equal(grid.detach().numpy(),a)


if __name__=='__main__': unittest.main()
