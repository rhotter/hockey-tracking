"""Check empty frames, assignment invariance, and box/score loss behavior."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from nbhl_sam3_losses import loss_for


class DetectionLossTests(unittest.TestCase):
    def output(self,boxes,logits):
        return {'pred_boxes':torch.tensor([boxes],requires_grad=True),
                'pred_logits':torch.tensor(logits).reshape(1,-1,1).requires_grad_(),
                'presence_logit_dec':torch.tensor([[10.]])}

    def test_correct_boxes_and_rejected_background_have_low_loss(self):
        target=torch.tensor([[.1,.2,.3,.6]])
        good=self.output([[.2,.4,.2,.4],[.8,.8,.1,.1]],[8.,-8.])
        bad=self.output([[.8,.8,.1,.1],[.7,.7,.1,.1]],[8.,8.])
        self.assertLess(float(loss_for(good,target)[0]),1e-5)
        self.assertGreater(float(loss_for(bad,target)[0]),1.)

    def test_empty_frame_has_finite_loss_and_suppresses_predictions(self):
        out=self.output([[.2,.4,.2,.4]],[2.])
        loss,_=loss_for(out,torch.empty((0,4)));loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(out['pred_logits'].grad[0,0,0]),0.)

    def test_matching_is_prediction_order_invariant(self):
        boxes=[[.2,.4,.2,.4],[.8,.8,.1,.1]];scores=[1.,-1.]
        target=torch.tensor([[.12,.22,.33,.61]])
        a,_=loss_for(self.output(boxes,scores),target)
        b,_=loss_for(self.output(boxes[::-1],scores[::-1]),target)
        self.assertAlmostEqual(float(a),float(b),places=6)


if __name__=='__main__':unittest.main()
