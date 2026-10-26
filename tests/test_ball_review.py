import unittest
from hockey_tracking.ball_review import apply_review, additional_visual_proposals, merge_reviewed, require_floor_support


class BallReviewTests(unittest.TestCase):
    def data(self):return {'frames':[{'frame':i,'state':'detected','pixel':[.2,.3],'xy':[.4,.5],'covariance':[[1,0],[0,1]]}for i in range(5)]}
    def test_review_cannot_create_positions_or_project_a_held_ball_onto_floor(self):
        data=self.data();data['frames'][4]={'frame':4,'state':'unknown'}
        ledger={'requests':{'0':{'status':'completed','frames':[0,1,2,3,4],'targets':[0,2,4],'answer':{'samples':[
            {'frame':0,'object':'ball','support':'held'}, {'frame':2,'object':'other','support':'unclear'}, {'frame':4,'object':'ball','support':'floor'}]}}}}
        out=apply_review(data,ledger)['frames']
        self.assertEqual(out[0]['pixel'],[.2,.3]);self.assertIsNone(out[0]['xy'])
        self.assertEqual(out[1]['review_frame'],0)
        self.assertEqual(out[2]['state'],'unknown');self.assertNotIn('pixel',out[2])
        self.assertEqual(out[4]['state'],'unknown')
        self.assertEqual(data['frames'][0]['xy'],[.4,.5])
    def test_failed_or_uncertain_review_withholds(self):
        ledger={'requests':{'0':{'status':'failed','frames':[0]},'1':{'status':'completed','frames':[1],'targets':[1],
            'answer':{'samples':[{'frame':1,'object':'uncertain','support':'unclear'}]}}}}
        self.assertTrue(all(r['state']=='unknown' for r in apply_review(self.data(),ledger)['frames']))
    def test_gap_cannot_survive_a_rejected_anchor(self):
        data=self.data();data['frames'][1].update(state='inferred',evidence_frames=[0,2])
        ledger={'requests':{'0':{'status':'completed','frames':[0,1,2],'targets':[0,1,2],
            'answer':{'samples':[{'frame':i,'object':'other' if i==2 else 'ball','support':'floor'}for i in range(3)]}}}}
        self.assertEqual(apply_review(data,ledger)['frames'][1]['state'],'unknown')
    def test_recall_can_only_add_new_visual_positions(self):
        primary=self.data();primary['frames'][1]={'frame':1,'state':'unknown'};primary['frames'][2]={'frame':2,'state':'unknown'}
        sensitive=self.data();sensitive['frames'][2].update(state='inferred')
        extra=additional_visual_proposals(primary,sensitive)
        self.assertEqual([r['state']for r in extra['frames']],['unknown','detected','unknown','unknown','unknown'])
        merged=merge_reviewed(primary,extra)
        self.assertEqual(merged['frames'][0],primary['frames'][0]);self.assertEqual(merged['frames'][1]['candidate_pass'],'recall')
        with self.assertRaises(ValueError):merge_reviewed(self.data(),extra)

    def test_weak_recovery_requires_reviewed_floor_and_valid_projection(self):
        data=self.data()
        for row,support in zip(data['frames'],['floor','airborne','held','unclear','floor']):row['review_support']=support
        data['frames'][4]['xy']=None
        result=require_floor_support(data)
        self.assertEqual([r['state'] for r in result['frames']],['detected','unknown','unknown','unknown','unknown'])
        self.assertEqual(result['frames'][0],data['frames'][0])
        self.assertEqual(data['frames'][1]['state'],'detected')


if __name__=='__main__':unittest.main()
