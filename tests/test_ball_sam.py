"""Guard against promoting invisible masks or weak conflicting tracks."""
import sys
from pathlib import Path
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]/'experiments/ball-sam-20261026'
sys.path.insert(0, str(EXPERIMENT))
from fusion import choose, choose_supported, eligible
from evaluate import accepted, predictions, selection
from hockey_tracking.ball_sequence import PointFusionConfig


def proposal(**updates):
    return dict(frame=10, seed_frame=5, pixel=[100.,100.], visible=True,
                inside_image=True, inside_crop=True, appearance_score=.6, **updates)


class BallSamTests(unittest.TestCase):
    def test_disagreement_needs_score_margin(self):
        tap=proposal(); sam={**tap,'pixel':[140.,100.],'appearance_score':.79}
        self.assertIs(choose(tap,sam),tap)
        sam['appearance_score']=.81
        self.assertIs(choose(tap,sam),sam)
        self.assertIs(choose(None,sam),sam)

    def test_agreement_keeps_one_position(self):
        tap=proposal(); sam={**tap,'pixel':[110.,100.],'appearance_score':.7}
        self.assertIs(choose(tap,sam),sam)
        sam['appearance_score']=.5
        self.assertIs(choose(tap,sam),tap)

    def test_consensus_requires_two_strong_distinct_seeds(self):
        tap={**proposal(),'appearance_score':.87}
        first={**proposal(),'pixel':[150.,100.],'appearance_score':.98}
        second={**first,'seed_frame':6,'pixel':[152.,100.],'appearance_score':.95}
        self.assertIs(choose_supported(tap,[first],'sam_consensus'),tap)
        self.assertIs(choose_supported(tap,[first,{**second,'seed_frame':5}],'sam_consensus'),tap)
        self.assertIs(choose_supported(tap,[first,{**second,'appearance_score':.89}],'sam_consensus'),tap)
        self.assertIs(choose_supported(tap,[first,{**second,'pixel':[180.,100.]}],'sam_consensus'),tap)
        self.assertIs(choose_supported(tap,[first,second],'sam_consensus'),first)
        self.assertIs(choose_supported(tap,[first,second],'conservative'),tap)

    def test_invisible_or_large_masks_cannot_become_observations(self):
        config={'appearance_threshold':.3}
        p=proposal(); self.assertTrue(accepted(p,'filtered',config))
        for flag in ['visible','inside_crop','inside_image','shape_valid']:
            self.assertFalse(accepted({**p,flag:False},'filtered',config))
        self.assertFalse(accepted({**p,'appearance_score':.29},'filtered',config))

    def test_context_or_strong_nearby_seed_required(self):
        p=proposal(); context={'frames':[{'hypotheses':[]}for _ in range(11)]}
        config=PointFusionConfig()
        self.assertFalse(eligible(p,10,context,config))
        self.assertTrue(eligible({**p,'appearance_score':.95},10,context,config))
        context['frames'][10]['hypotheses']=[{'pixel':[100/1920,100/1080],'radius_px':60}]
        self.assertTrue(eligible(p,10,context,config))
        self.assertFalse(eligible({**p,'shape_valid':False},10,context,config))

    def test_nearest_seed_chosen_before_visibility(self):
        far=proposal(); near={**far,'visible':False}
        result=predictions({'tracks':[{'seed_frame':1,'rows':[far]},{'seed_frame':9,'rows':[near]}]})
        self.assertEqual(result[10]['seed_frame'],9)
        self.assertFalse(result[10]['visible'])

    def test_validation_does_not_select_configuration(self):
        def item(dev,val):
            return {'scores':{'filtered':{'development':{'selection_utility':dev,'within_12px':dev,'outside_25px':0},'validation':{'selection_utility':val}}}}
        self.assertEqual(selection({'sam-a':item(2,100),'sam-b':item(3,-100)}),'sam-b')


if __name__=='__main__':unittest.main()
