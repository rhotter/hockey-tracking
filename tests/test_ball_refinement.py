from copy import deepcopy
import unittest
from hockey_tracking.ball_refinement import refine_sequence, RefinementConfig


class BallRefinementTests(unittest.TestCase):
    def setUp(self):
        self.n = 9
        self.rink = {'frames':[{'frame':f,'camera_valid':False}for f in range(self.n)],
                     'relative_bounds':[0,1920,0,1080]}
        self.baseline = {'frames':[{'frame':f,'state':'context','pixel':[.5,.5]}for f in range(self.n)]}
        self.detector = [{'frame':f,'candidates':[]}for f in range(self.n)]

    def stream(self, frames, batch='test'):
        def pixel(f): return [100+2*f,200]
        seeds = {'batch':batch,'seeds':[{'frame':f,'pixel':pixel(f),'support':'floor',
                 'appearance_score':.01,'evidence':'Visible model sighting.'}for f in frames]}
        tracks = {'complete':True,'protocol':{'batch':batch,'offload_state_to_cpu':False},'tracks':[]}
        for sf in frames:
            rows = [{'frame':f,'pixel':pixel(f),'visible':True,'shape_valid':True,'inside_image':True,
                     'inside_crop':True,'appearance_score':.01}for f in range(self.n)]
            tracks['tracks'].append({'seed_frame':sf,'seed_pixel':pixel(sf),'rows':rows})
        return seeds,tracks

    def run_fusion(self, seeds, tracks, **kwargs):
        return refine_sequence(self.rink,self.baseline,seeds,tracks,self.detector,**kwargs)

    def test_two_sightings_support_intervening_tracks_without_relabeling_context(self):
        seeds,tracks = self.stream([0,6])
        result,report = self.run_fusion([seeds],[tracks])
        self.assertEqual(report['accepted_seeds'],2)
        self.assertEqual(result['frames'][3]['state'],'tracked')
        self.assertEqual(result['frames'][3]['temporal_votes'],2)
        self.assertEqual(result['frames'][8]['state'],'context')

    def test_two_copies_of_one_seed_are_only_one_temporal_vote(self):
        a,ta = self.stream([0],'a');b,tb = self.stream([0],'b')
        self.detector[0]['candidates']=[{'pixel':[100,200],'score':.9,'appearance_score':.9}]
        result,report = self.run_fusion([a,b],[ta,tb])
        self.assertEqual(report['accepted_seeds'],2)
        self.assertEqual(result['frames'][3]['state'],'context')

    def test_bad_seed_mask_is_rejected_even_with_detector_support(self):
        seeds,tracks = self.stream([0])
        tracks['tracks'][0]['rows'][0]['pixel']=[500,200]
        self.detector[0]['candidates']=[{'pixel':[100,200],'score':.9,'appearance_score':.9}]
        result,report = self.run_fusion([seeds],[tracks])
        self.assertEqual(report['accepted_seeds'],0)
        self.assertEqual(result['frames'][0]['state'],'context')

    def test_one_way_agreement_is_not_a_cycle(self):
        seeds,tracks = self.stream([0,6])
        tracks['tracks'][1]['rows'][0]['pixel']=[500,200]
        _,report = self.run_fusion([seeds],[tracks])
        self.assertEqual(report['accepted_seeds'],0)

    def test_edit_blocks_cross_cut_links_and_outputs(self):
        seeds,tracks = self.stream([0,6])
        result,report = self.run_fusion([seeds],[tracks],excluded_frames=[3])
        self.assertEqual(report['accepted_seeds'],0)
        self.assertEqual(result['frames'][3]['state'],'unknown')

    def test_unvalidated_cpu_state_is_not_promoted(self):
        seeds,tracks = self.stream([0,6]);tracks['protocol']['offload_state_to_cpu']=True
        with self.assertRaisesRegex(ValueError,'GPU-state'):
            self.run_fusion([seeds],[tracks])

    def test_original_detection_is_preserved(self):
        seeds,tracks = self.stream([0,6])
        self.baseline['frames'][3]={'frame':3,'state':'detected','pixel':[.6,.4]}
        result,_ = self.run_fusion([seeds],[tracks])
        self.assertEqual(result['frames'][3],self.baseline['frames'][3])

    def test_weak_appearance_does_not_promote_consistent_wrong_object(self):
        seeds,tracks = self.stream([0,6])
        result,_ = self.run_fusion([seeds],[tracks],config=RefinementConfig(minimum_output_appearance=.02))
        self.assertEqual(result['frames'][3],self.baseline['frames'][3])

    def test_held_object_is_kept_contextual(self):
        seeds,tracks = self.stream([0,6])
        for seed in seeds['seeds']: seed['support']='held'
        result,_ = self.run_fusion([seeds],[tracks],config=RefinementConfig(held_as_context=True))
        self.assertEqual(result['frames'][0]['state'],'context')
        self.assertEqual(result['frames'][3]['state'],'context')
        self.assertEqual(result['frames'][3]['review_support'],'held')

    def test_interpolation_preserves_supported_held_ball(self):
        seeds,tracks = self.stream([0,3,6])
        seeds['seeds'][1]['support']='held'
        result,_ = self.run_fusion([seeds],[tracks],config=RefinementConfig(held_as_context=True))
        self.assertEqual(result['frames'][3]['state'],'context')
        self.assertEqual(result['frames'][3]['review_support'],'held')
        self.assertEqual(result['frames'][3]['reason'],'sequence_sighting_with_sam_corroboration')

    def test_duplicate_track_stream_is_rejected(self):
        seeds,tracks = self.stream([0,6])
        with self.assertRaisesRegex(ValueError,'Duplicate tracking seed'):
            self.run_fusion([seeds],[tracks,tracks])

    def test_detector_seed_cannot_propagate_across_cut(self):
        seeds,tracks = self.stream([0])
        for f in [0,2]:
            self.detector[f]['candidates']=[{'pixel':[100+2*f,200],'score':.9,'appearance_score':.9}]
        result,_ = self.run_fusion([seeds],[tracks],excluded_frames=[1])
        self.assertEqual(result['frames'][0]['state'],'detected')
        self.assertEqual(result['frames'][2],self.baseline['frames'][2])

    def isolated_spike(self, neighbor_state='detected'):
        for row in self.rink['frames']:
            row.update(camera_valid=True,image_to_floor=[[1,0,0],[0,1,0],[0,0,1]])
        seeds,tracks=self.stream([4])
        self.detector[4]['candidates']=[{'pixel':[108,200],'score':.9,'appearance_score':.9}]
        for f in [3,5]:
            self.baseline['frames'][f]={'frame':f,'state':neighbor_state,'pixel':[.6,.5]}
        return seeds,tracks

    def test_isolated_new_detection_is_replaced_by_explicit_inference(self):
        seeds,tracks=self.isolated_spike()
        result,report=self.run_fusion([seeds],[tracks],config=RefinementConfig(reject_isolated_spikes=True))
        self.assertEqual([r['frame'] for r in report['isolated_observation_rejections']],[4])
        # Old anchors are not changed. Their short gap is an estimate, not a detection.
        self.assertEqual(result['frames'][3],self.baseline['frames'][3])
        self.assertEqual(result['frames'][4]['state'],'inferred')
        self.assertEqual(result['frames'][4]['pixel'],[.6,.5])

    def test_context_neighbors_cannot_reject_a_visual_ball(self):
        seeds,tracks=self.isolated_spike('context')
        result,report=self.run_fusion([seeds],[tracks],config=RefinementConfig(reject_isolated_spikes=True))
        self.assertEqual(report['isolated_observation_rejections'],[])
        self.assertEqual(result['frames'][4]['state'],'detected')

    def test_missing_camera_alignment_disables_spike_rejection(self):
        seeds,tracks=self.isolated_spike();self.rink['frames'][4]['camera_valid']=False
        result,report=self.run_fusion([seeds],[tracks],config=RefinementConfig(reject_isolated_spikes=True))
        self.assertEqual(report['isolated_observation_rejections'],[])
        self.assertEqual(result['frames'][4]['state'],'detected')

    def test_fast_coherent_motion_is_not_an_isolated_spike(self):
        seeds,tracks=self.isolated_spike()
        seeds['seeds'][0]['pixel']=[1000,200]
        tracks['tracks'][0]['seed_pixel']=[1000,200]
        tracks['tracks'][0]['rows'][4]['pixel']=[1000,200]
        self.detector[4]['candidates'][0]['pixel']=[1000,200]
        self.baseline['frames'][3]['pixel']=[600/1920,200/1080]
        self.baseline['frames'][5]['pixel']=[1400/1920,200/1080]
        result,report=self.run_fusion([seeds],[tracks],config=RefinementConfig(reject_isolated_spikes=True))
        self.assertEqual(report['isolated_observation_rejections'],[])
        self.assertEqual(result['frames'][4]['state'],'detected')


if __name__ == '__main__': unittest.main()
