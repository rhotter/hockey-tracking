import unittest
import numpy as np
from hockey_tracking.ball_path import select_path, fuse_candidates, filter_person_parts, filter_verified, PathConfig
from hockey_tracking.ball_track import make_replay


def box(x,y=500,score=.6):return [x-5,y-5,x+5,y+5,score,0]


def replay(n,homographies=None):
    return {'relative_bounds':[-2,2,-1,1],'frames':[
        {'objects':[],'image_to_floor':homographies[i] if homographies else None} for i in range(n)]}


class BallPathTests(unittest.TestCase):
    def test_prefers_motion_path_to_one_frame_distractor(self):
        rows=[{'detections':[box(100+i*10)]} for i in range(9)]
        rows[4]['detections'].append(box(900,score=.99))
        out=select_path(rows,replay(9))['detections']
        np.testing.assert_allclose([(b[0]+b[2])/2 for b in out],np.arange(9)*10+100)

    def test_camera_pan_does_not_break_a_stationary_floor_ball(self):
        # Both image positions refer to the same floor point after a 300px pan.
        hs=[np.eye(3).tolist(),[[1,0,-300],[0,1,0],[0,0,1]]]
        r=replay(2,hs);r['relative_bounds']=[-1000,1000,-1000,1000]
        out=select_path([{'detections':[box(100)]},{'detections':[box(400)]}],r,
                        PathConfig(birth_cost=2.5))['detections']
        self.assertTrue(all(b is not None for b in out))

    def test_cut_is_unknown_even_with_high_score(self):
        rows=[{'detections':[box(100,score=.99)]} for _ in range(5)]
        out=select_path(rows,replay(5),excluded_frames=[2])['detections']
        self.assertIsNone(out[2]);self.assertIsNotNone(out[1]);self.assertIsNotNone(out[3])

    def test_image_tracking_survives_missing_floor_calibration(self):
        out=select_path([{'detections':[box(100)]} for _ in range(4)],replay(4))
        self.assertTrue(all(b is not None for b in out['detections']))

    def test_empty_observations_do_not_invent_positions(self):
        self.assertEqual(select_path([{'detections':[]}]*4,replay(4))['detections'],[None]*4)
        self.assertEqual(select_path([],replay(0))['detections'],[])

    def test_duplicate_candidates_do_not_inflate_agreement(self):
        a=[{'detections':[box(100),box(101),box(102)]}]
        b=[{'detections':[box(103)]}]
        only=fuse_candidates([a])[0]['detections'];both=fuse_candidates([a,b])[0]['detections']
        self.assertEqual(len(only),1);self.assertAlmostEqual(only[0][4],.6)
        self.assertEqual(len(both),1);self.assertAlmostEqual(both[0][4],.78)

    def test_far_apart_views_remain_alternatives(self):
        result=fuse_candidates([[{'detections':[box(100)]}],[{'detections':[box(800)]}]])
        self.assertEqual(len(result[0]['detections']),2)

    def test_joint_rejection_gates_preserve_candidate_alignment(self):
        rows=[{'frame':0,'detections':[box(100),box(200),box(300),box(400)]}]
        verified=[{'frame':0,'probabilities':[.9,.9,.01,.9]}]
        markings=[{'features':[{'extended_marking':True},{'extended_marking':False},{'extended_marking':False},{'extended_marking':False,'elongated_structure':True}]}]
        self.assertEqual(filter_verified(rows,verified,.1,markings)[0]['detections'],[box(200)])
        with self.assertRaises(ValueError):filter_verified(rows,[{'probabilities':[.9]}],.1)

    def test_person_context_rejects_head_and_shoe_but_keeps_nearby_floor_ball(self):
        joints=[[0,0,0] for _ in range(17)];joints[15]=[100,480,.9]
        people=[{'boxes':[[60,200,140,510,.9,0]],'keypoints':[joints]}]
        detections=[{'detections':[box(100,220),box(100,488),box(170,500)]}]
        result=filter_person_parts(detections,people)[0]['detections']
        self.assertEqual(result,[box(170,500)])

    def test_rendering_keeps_image_position_without_inventing_rink_position(self):
        r=replay(2);r.update(fps=15,duration_s=2/15,source_id='test',start_s=0)
        for i,row in enumerate(r['frames']):row.update(frame=i,camera_valid=False)
        result=make_replay(r,[box(100),None])['frames']
        self.assertEqual(result[0]['state'],'detected');self.assertIsNone(result[0]['xy'])
        np.testing.assert_allclose(result[0]['pixel'],[100/1920,500/1080])
        self.assertEqual(result[1]['state'],'unknown')


if __name__=='__main__':unittest.main()
