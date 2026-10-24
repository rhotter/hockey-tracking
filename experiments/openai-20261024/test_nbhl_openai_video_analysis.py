import unittest
from analyze_nbhl_openai_video_baseline import score,direct_frames


class ScoringTests(unittest.TestCase):
    def reference(self):
        return {'observations':[{'id':'a','clip':'c','frame':0,'bbox':[0,0,10,20]},
                                {'id':'b','clip':'c','frame':15,'bbox':[20,0,30,20]},
                                {'id':'c','clip':'c','frame':15,'bbox':[40,0,50,20]}],
                'pairs':[{'a':'a','b':'b','clip':'c','truth':'same'},
                         {'a':'a','b':'c','clip':'c','truth':'different'}]}

    def test_missing_player_is_not_a_correct_rejection(self):
        result=score(self.reference(),'c',{0:[{'track_id':'1','bbox':[0,0,10,20]}]})
        self.assertEqual(result['counts']['missing'],2)
        self.assertEqual(result['counts']['correct'],0)
        self.assertIsNone(result['false_positives'])

    def test_identity_swap_counts_wrong(self):
        result=score(self.reference(),'c',{0:[{'track_id':'1','bbox':[0,0,10,20]}],
                15:[{'track_id':'2','bbox':[20,0,30,20]},{'track_id':'1','bbox':[40,0,50,20]}]})
        self.assertEqual(result['counts']['matched_observations'],3)
        self.assertEqual(result['counts']['correct'],0)
        self.assertEqual(result['counts']['wrong'],1)

    def test_duplicate_id_in_frame_cannot_score(self):
        result=score(self.reference(),'c',{15:[{'track_id':'1','bbox':[20,0,30,20]},
                                                   {'track_id':'1','bbox':[40,0,50,20]}]})
        self.assertEqual(result['counts']['matched_observations'],0)

    def test_normalized_coordinates_use_source_size(self):
        record={'parsed':{'players':[{'player_id':'1'}],'frames':[{'frame':0,'objects':[
            {'player_id':'1','bbox':[100,200,300,400]}]}]}}
        clip={'output_frames':[0],'images':[{'frame':0,'source_dimensions':[1920,1080]}]}
        self.assertEqual(direct_frames(record,clip)[0][0]['bbox'],[192,216,576,432])


if __name__=='__main__':unittest.main()
