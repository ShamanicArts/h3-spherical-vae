import math,unittest
import torch
from final_shift import build_final_shift_euler,roll_video,accept_edge_columns

class FinalShiftTest(unittest.TestCase):
    shapes=((1,24,2,2,8),(1,32,2,3))
    def state(self):
        return torch.arange(sum(math.prod(s[1:]) for s in self.shapes),dtype=torch.float32).reshape(1,1,-1)/100
    def test_roll_roundtrip_and_audio(self):
        x=self.state();r=roll_video(x,4,self.shapes);n=math.prod(self.shapes[0][1:])
        self.assertTrue(torch.equal(x,roll_video(r,-4,self.shapes)))
        self.assertTrue(torch.equal(x[...,n:],r[...,n:]))
    def test_full_schedules_and_final_footprint(self):
        for steps in (50,100):
            calls=[];observed=[];rows=[];x=self.state()
            # A deliberately chart-dependent fake predictor exposes roll/unroll mistakes.
            def model(value,sigma,**kw):calls.append(value.clone());return .3*value+torch.arange(value.numel()).reshape_as(value).to(value)/1000
            sigmas=torch.linspace(1,0,steps+1)
            out=build_final_shift_euler(self.shapes,receipt=rows,final_observer=lambda *a:observed.append(a))(model,x.clone(),sigmas)
            self.assertEqual(len(calls),steps+1);self.assertEqual(sum(r['model_evaluations'] for r in rows),steps+1)
            state,canonical,corrected,control,treated=observed[0];n=math.prod(self.shapes[0][1:])
            self.assertTrue(torch.equal(out,treated));self.assertTrue(torch.equal(control[...,n:],treated[...,n:]))
            a,b=(v[...,:n].reshape(self.shapes[0]) for v in (control,treated))
            self.assertTrue(torch.equal(a[...,1:-1],b[...,1:-1]));self.assertFalse(torch.equal(a,b))
            self.assertTrue(torch.equal(calls[-1],roll_video(calls[-2],4,self.shapes)))
            # Unchanged trajectory and same final canonical prediction.
            baseline=build_final_shift_euler(self.shapes,enabled=False)(model,x.clone(),sigmas)
            self.assertTrue(torch.equal(baseline,control))
    def test_reject_masks_and_bad_schedule(self):
        sample=build_final_shift_euler(self.shapes);x=self.state();model=lambda x,s,**kw:x
        for sigmas in (torch.tensor([1.,.5,.5,0]),torch.tensor([1.,.1]),torch.tensor([1.,float('nan'),0.])):
            with self.assertRaises(ValueError):sample(model,x,sigmas)
        with self.assertRaises(ValueError):sample(model,x,torch.tensor([1.,0.]),extra_args={'denoise_mask':torch.ones(1)})
    def test_bad_shape(self):
        with self.assertRaises(ValueError):build_final_shift_euler(((1,24,2,2,6),(1,32,2,3)))

if __name__=='__main__':unittest.main()
