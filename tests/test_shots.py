import unittest
from copy import deepcopy
import importlib.util
from pathlib import Path
from hockey_tracking.shots import classify_event,combine_events,event_totals,validate_event

class ShotTests(unittest.TestCase):
    def setUp(self):
        self.event={'release_frame':100,'outcome_frame':110,'team_color':'red','shooter_jersey':None,
          'shooter_xy_1000':[500,500],'net_xy_1000':[700,400],'attempt':'clear','outcome':'save',
          'on_target':'yes','evidence':{'release_visible':True,'ball_to_goal_visible':True,'goalie_save_visible':True,
          'goal_crossing_visible':False,'score_change_visible':False,'official_goal_signal_visible':False,
          'rebound_release_visible':False},'evidence_frames':[100,110],'explanation':'On-target shot stopped by goalkeeper.'}
        self.clip={'id':'demo','source_id':'test','frames':1000,'start_s':900}
    def record(self,events,id='window1'):
        return {'id':id,'status':'completed','core':[0,300],'sample_frames':list(range(300)),'answer':{'events':events}}
    def test_on_target_save_counts(self):self.assertEqual(classify_event(self.event)[0],'sog')
    def test_goalkeeper_motion_alone_does_not_count(self):
        self.event['evidence']['ball_to_goal_visible']=False
        self.assertEqual(classify_event(self.event)[0],'needs_review')
    def test_wide_goalkeeper_collection_does_not_count(self):
        self.event['on_target']='no'
        self.assertEqual(classify_event(self.event)[0],'needs_review')
    def test_blocks_misses_and_posts_are_not_sog(self):
        for outcome in ['blocked','miss','post','not_shot','disallowed_goal']:
            self.event['outcome']=outcome;self.assertEqual(classify_event(self.event)[0],'not_sog')
    def test_celebration_without_goal_evidence_does_not_count(self):
        self.event['outcome']='goal';self.assertEqual(classify_event(self.event)[0],'needs_review')
        self.event['evidence']['official_goal_signal_visible']=True
        self.assertEqual(classify_event(self.event)[0],'needs_review')
        self.event['evidence']['score_change_visible']=True
        self.assertEqual(classify_event(self.event)[0],'sog')
    def test_two_rebound_releases_in_one_window_stay_separate(self):
        second=deepcopy(self.event);second.update(release_frame=104,outcome_frame=114)
        result=combine_events([self.record([self.event,second])],self.clip)
        self.assertEqual(len(result),2);self.assertEqual(event_totals(result)['sog'],2)
    def test_cross_window_duplicates_count_once(self):
        second=deepcopy(self.event);second.update(release_frame=102,outcome_frame=112)
        result=combine_events([self.record([self.event]),self.record([second],'window2')],self.clip)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['recording_seconds'],900+100/15)
    def test_conflicting_overlap_is_not_counted(self):
        second=deepcopy(self.event);second['outcome']='blocked'
        result=combine_events([self.record([self.event]),self.record([second],'window2')],self.clip)
        self.assertEqual(result[0]['automatic_status'],'needs_review')
    def test_incomplete_window_is_not_a_zero_shot_observation(self):
        r=self.record([self.event]);r['status']='incomplete'
        self.assertEqual(combine_events([r],self.clip),[])
    def test_unseen_evidence_frame_is_rejected(self):
        self.event['evidence_frames']=[500]
        with self.assertRaises(ValueError):combine_events([self.record([self.event])],self.clip)
    def test_invalid_time_and_coordinates_are_rejected(self):
        self.event['outcome_frame']=99
        with self.assertRaises(ValueError):validate_event(self.event,1000)
        self.event['outcome_frame']=110;self.event['net_xy_1000']=[float('nan'),0]
        with self.assertRaises(ValueError):validate_event(self.event,1000)
    def test_unknown_team_not_assigned_to_opponent(self):
        self.event['team_color']='unknown'
        result=event_totals(combine_events([self.record([self.event])],self.clip))
        self.assertEqual(result['teams']['unknown']['sog'],1);self.assertNotIn('red',result['teams'])
    def test_contextual_save_needs_corroboration(self):
        self.event.update(basis='contextual',confidence='high',alternative_explanation=None,
                          cues=['shooting_motion','goalkeeper_interception'])
        self.event['evidence']['ball_to_goal_visible']=False
        self.assertEqual(classify_event(self.event)[0],'needs_review')
        self.event['cues'].append('defensive_zone_restart')
        self.assertEqual(classify_event(self.event)[0],'sog')
        self.event['alternative_explanation']='Could be a wide centering pass.'
        self.assertEqual(classify_event(self.event)[0],'needs_review')
    def test_medium_confidence_direct_claim_is_withheld(self):
        self.event['confidence']='medium'
        self.assertEqual(classify_event(self.event)[0],'needs_review')
    def test_score_change_plus_center_restart_establishes_goal_without_shooter(self):
        self.event.update(outcome='goal',attempt='possible',basis='contextual',confidence='high',cues=['score_change','center_restart'])
        self.event['evidence']['score_change_visible']=True
        self.assertEqual(classify_event(self.event)[0],'sog')
        self.event['cues']=['center_restart']
        self.event['evidence']['score_change_visible']=False
        self.assertEqual(classify_event(self.event)[0],'needs_review')

    def test_failed_context_uses_completed_verifier_and_reports_gap(self):
        path=Path(__file__).resolve().parents[1]/'scripts/detect_shots.py'
        spec=importlib.util.spec_from_file_location('detect_shots',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        requests=[{'id':'a-verify-0000'},{'id':'b-verify-0000'},{'id':'c-verify-0000'}]
        verify={'a-verify-0000':self.record([self.event]),'b-verify-0000':self.record([self.event])}
        context={'a-context-0000':{'status':'failed'},'b-context-0000':{'status':'completed','answer':{'events':[]}}}
        selected,missing,fallback=module.select_records(requests,verify,context)
        self.assertEqual(len(selected),2)
        self.assertEqual(selected[1]['answer']['events'],[])
        self.assertEqual(missing,['c-verify-0000'])
        self.assertEqual(fallback,['a-verify-0000'])
        recovered={'c-context-recovery-0000':{'status':'completed','answer':{'events':[self.event]}}}
        selected,missing,fallback=module.select_records(requests,verify,context,recovered)
        self.assertEqual(len(selected),3);self.assertEqual(missing,[])
        self.assertEqual(selected[-1]['answer']['events'],[self.event])

if __name__=='__main__':unittest.main()
