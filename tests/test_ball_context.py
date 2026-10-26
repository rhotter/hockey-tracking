import unittest
import numpy as np
from hockey_tracking.ball_context import Config, infer


def replay(anchors=(), count=30, occluded=False, cuts=()):
    rows=[]
    for f in range(count):
        xy=[.2+.001*f,.5]
        rows.append({'frame':f,'camera_valid':f not in cuts,
                     'image_to_floor':[[1/1920,0,0],[0,1/1080,0],[0,0,1]] if f not in cuts else None,
                     'objects':[{'track_id':'T1','bbox':[0,0,1,1]}] if occluded else [],
                     'ball':{'xy':xy,'pixel':xy,'confidence':.8} if f in anchors else None})
    return {'frames':rows,'fps':15,'duration_s':count/15,'source_id':'synthetic',
            'start_s':0,'relative_bounds':[0,1,0,1]}


class BallContextTests(unittest.TestCase):
    def test_no_detection_no_hallucination(self):
        result=infer(replay(occluded=True))
        self.assertTrue(all(r['state']=='unknown' for r in result['frames']))

    def test_isolated_anchor_cannot_define_velocity(self):
        result=infer(replay([2]))['frames']
        self.assertEqual(sum(r['state']=='detected' for r in result),1)
        self.assertFalse(any(r['state']=='inferred' for r in result))

    def test_uncertainty_grows_then_prediction_expires(self):
        rows=infer(replay([0,1]),config=Config(bridge=False))['frames']
        radii=[rows[f]['model_radius_px'] for f in [1,2,3,4]]
        self.assertTrue(all(a<b for a,b in zip(radii,radii[1:])))
        self.assertEqual(rows[5]['state'],'unknown')

    def test_future_observation_bridges_short_gap(self):
        rows=infer(replay([0,1,7,8]),config=Config(coast=False))['frames']
        self.assertTrue(all(rows[f]['state']=='inferred' for f in range(2,7)))
        for f in range(2,7):np.testing.assert_allclose(rows[f]['xy'],[.2+.001*f,.5])
        self.assertGreater(rows[4]['model_radius_px'],rows[2]['model_radius_px'])

    def test_occlusion_extends_gap_without_inventing_possession(self):
        r=replay([0,1,11,12],occluded=True)
        context=infer(r,config=Config(coast=False))['frames']
        motion=infer(r,config=Config(coast=False,use_occlusion=False))['frames']
        self.assertEqual(context[6]['reason'],'occlusion_bridge')
        self.assertEqual(motion[6]['state'],'unknown')
        np.testing.assert_allclose(context[6]['xy'],[.206,.5])
        self.assertNotIn('possession',context[6])

    def test_no_bridge_across_cut_or_long_gap(self):
        cut=infer(replay([0,1,5,6],cuts=[3]),config=Config(coast=False))['frames']
        self.assertEqual(cut[2]['state'],'unknown')
        self.assertEqual(cut[3]['reason'],'camera_unavailable')
        long=infer(replay([0,1,20,21],occluded=True))['frames']
        self.assertEqual(long[10]['state'],'unknown')

    def test_contradictory_future_anchor_blocks_prediction(self):
        r=replay([0,1,5,6]);r['frames'][5]['ball']['xy']=[.99,.99]
        r['frames'][5]['ball']['pixel']=[.99,.99]
        result=infer(r)['frames']
        self.assertEqual(result[2]['state'],'unknown')

    def test_overwide_uncertainty_is_withheld(self):
        rows=infer(replay([0,1]),config=Config(max_radius_px=13))['frames']
        self.assertEqual(rows[2]['state'],'unknown')

    def test_covariance_is_positive_and_conditional(self):
        r=infer(replay([0,1,7,8]))
        self.assertFalse(r['uncertainty_calibrated'])
        self.assertTrue(r['floor_projection_assumption'])
        for f in r['frames']:
            if f['state']=='unknown':continue
            self.assertTrue(np.all(np.linalg.eigvalsh(f['covariance'])>0))
            self.assertTrue(np.all(np.linalg.eigvalsh(f['pixel_covariance'])>0))


if __name__=='__main__':unittest.main()
