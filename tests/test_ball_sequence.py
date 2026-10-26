import unittest
from copy import deepcopy
import numpy as np
from hockey_tracking.ball_sequence import build_context_replay,merge_visual_and_context,fuse_point_tracks


class BallSequenceTests(unittest.TestCase):
    def fixture(self):
        frames=[]
        for f in range(9):
            x=170 if f==4 else 100+10*f
            frames.append({'frame':f,'camera_valid':True,'image_to_floor':np.eye(3).tolist(),
                           'objects':[{'track_id':'1','foot':[x/1920,200/1080],'bbox':[0,0,1,1]}]})
        replay={'source_id':'test','start_s':0,'duration_s':.6,'fps':15,'relative_bounds':[0,1920,0,1080],'frames':frames}
        def sample(f,x):
            return {'frame':f,'play_state':'carried','visibility':'occluded','unknown_weight':.1,'transition_evidence':'Pass received, then same player stickhandles.',
                    'hypotheses':[{'xy_1000':[(x+30)/1.92,180/1.08],'radius_px':60,'weight':.9,'carrier_track_id':'T1','support':'floor','reason':'near stick blade'}]}
        ledger={'requests':{'0000':{'status':'completed','targets':[0,8],'answer':{'samples':[sample(0,100),sample(8,180)]}}}}
        return replay,ledger

    def test_hidden_ball_follows_observed_carrier_motion(self):
        replay,ledger=self.fixture();out=build_context_replay(replay,ledger)
        mid=out['frames'][4]
        np.testing.assert_allclose(np.array(mid['pixel'])*[1920,1080],[200,180])
        self.assertEqual(mid['hypotheses'][0]['motion_source'],'carrier_motion')
        self.assertEqual(mid['state'],'context');self.assertEqual(mid['visibility'],'occluded')
        self.assertFalse(out['uncertainty_calibrated'])

    def test_edits_break_context_and_held_balls_have_no_floor_position(self):
        replay,ledger=self.fixture()
        for row in ledger['requests']['0000']['answer']['samples']:row['hypotheses'][0]['support']='held'
        out=build_context_replay(replay,ledger,excluded_frames=[4])
        self.assertIsNone(out['frames'][0]['xy'])
        self.assertTrue(all(out['frames'][f]['state']=='unknown'for f in range(1,8)))

    def test_weak_context_does_not_fill_every_frame(self):
        replay,ledger=self.fixture()
        for row in ledger['requests']['0000']['answer']['samples']:
            row['unknown_weight']=.8;row['hypotheses'][0]['weight']=.2
        self.assertTrue(all(r['state']=='unknown'for r in build_context_replay(replay,ledger)['frames']))

    def test_visual_observations_take_precedence_without_mutating_inputs(self):
        replay,ledger=self.fixture();context=build_context_replay(replay,ledger);visual=deepcopy(context)
        visual['frames']=[{'frame':f,'state':'unknown'}for f in range(9)]
        visual['frames'][4]={'frame':4,'state':'detected','pixel':[.5,.5]}
        merged=merge_visual_and_context(visual,context)
        self.assertEqual(merged['frames'][4],visual['frames'][4]);self.assertEqual(merged['frames'][3]['state'],'context')
        self.assertEqual(context['frames'][4]['state'],'context')

    def test_point_fusion_requires_video_visibility_context_and_temporal_support(self):
        replay,ledger=self.fixture();context=build_context_replay(replay,ledger)
        visual={'frames':[{'frame':f,'state':'unknown'}for f in range(9)]}
        verified={'frames':[{'frame':f,'proposals':[]}for f in range(9)]}
        for f in [0,3,4,5,8]:
            verified['frames'][f]['proposals']=[{'frame':f,'pixel':(np.array(context['frames'][f]['pixel'])*[1920,1080]).tolist(),
                'visible':True,'inside_image':True,'inside_crop':True,'appearance_score':.5,'seed_frame':4}]
        fused=fuse_point_tracks(replay,visual,context,verified)
        self.assertEqual([r['frame']for r in fused['frames']if r['state']=='tracked'],[3,4,5])
        self.assertTrue(fused['frames'][4]['context_agreement'])
        verified['frames'][4]['proposals'][0]['visible']=False
        self.assertFalse(any(r['state']=='tracked'for r in fuse_point_tracks(replay,visual,context,verified)['frames']))

    def test_point_fusion_does_not_invent_floor_support_without_context(self):
        replay,ledger=self.fixture();context=build_context_replay(replay,ledger)
        context['frames']=[{'frame':f,'state':'unknown'}for f in range(9)]
        visual=deepcopy(context);verified={'frames':[{'frame':f,'proposals':[]}for f in range(9)]}
        for f in [3,4,5]:
            verified['frames'][f]['proposals']=[{'frame':f,'pixel':[200,180],'visible':True,'inside_image':True,'inside_crop':True,'appearance_score':.95,'seed_frame':4}]
        fused=fuse_point_tracks(replay,visual,context,verified)
        self.assertEqual(fused['frames'][4]['state'],'tracked');self.assertIsNone(fused['frames'][4]['xy'])
        self.assertEqual(fused['frames'][4]['review_support'],'unclear')

    def test_camera_pan_preserves_the_same_floor_location(self):
        replay,ledger=self.fixture()
        for f,row in enumerate(replay['frames']):
            row['image_to_floor']=[[1,0,-10*f],[0,1,0],[0,0,1]]
        for row in ledger['requests']['0000']['answer']['samples']:
            row['hypotheses'][0]['carrier_track_id']=None
        out=build_context_replay(replay,ledger)
        np.testing.assert_allclose(np.array(out['frames'][4]['pixel'])*[1920,1080],[170,180])
        np.testing.assert_allclose(out['frames'][0]['xy'],out['frames'][4]['xy'])
        np.testing.assert_allclose(out['frames'][4]['xy'],out['frames'][8]['xy'])

    def test_contested_locations_remain_separate_hypotheses(self):
        replay,ledger=self.fixture()
        for row in ledger['requests']['0000']['answer']['samples']:
            row['play_state']='contested';row['unknown_weight']=.15
            h=row['hypotheses'][0];h['weight']=.55;h['carrier_track_id']=None
            other=deepcopy(h);other['xy_1000'][0]+=200;other['weight']=.3
            row['hypotheses'].append(other)
        mid=build_context_replay(replay,ledger)['frames'][4]
        self.assertEqual(len(mid['hypotheses']),2)
        np.testing.assert_allclose(np.array(mid['hypotheses'][1]['pixel'])-mid['hypotheses'][0]['pixel'],[.2,0])
        np.testing.assert_allclose(mid['pixel'],mid['hypotheses'][0]['pixel'])


if __name__=='__main__':unittest.main()
