"""Behavioral checks for label-free SAM3 association constraints."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from nbhl_sam3_hybrid import matches,split_appearance,consensus_edges


class HybridTests(unittest.TestCase):
    def test_ambiguous_duplicate_cannot_claim_player(self):
        box=[0,0,10,20]
        self.assertEqual(matches([box,box],[box]),[])
        self.assertEqual(matches([box],[box,box]),[])
        self.assertEqual(matches([box],[[100,0,110,20]]),[])

    def test_appearance_split_includes_unclear_intermediate_frames(self):
        frames=[];raw=[]
        for f in range(5):
            confidence=.2 if f==2 else .9
            frames.append({'scene':0,'detections':[[0,0,10,20,confidence]]})
            raw.append({'frame':f,'objects':[{'track_id':'A','detection_index':0,'bbox':[0,0,10,20],'confidence':confidence}]})
        features={'features':np.asarray([[1,0]]+[[0,1]]*4,dtype=np.float32),'offsets':np.arange(6)}
        result=split_appearance({'frames':frames},features,{'frames':raw},.6)
        ids=[r['objects'][0]['track_id'] for r in result['frames']]
        self.assertEqual(ids,['A/s0']+['A/s1']*4)
        self.assertTrue(all(r['objects'][0]['track_id']=='A' for r in raw))

    def test_consensus_requires_fresh_repeated_evidence(self):
        edge={'a':'A','b':'B','cosine_mean':.72}
        vote={'id':'X','support':8,'fraction':1.,'distance_to_endpoint':0}
        votes={'A':{'end':vote},'B':{'start':vote}}
        self.assertAlmostEqual(consensus_edges([edge],votes,bonus=.05)[0]['cosine_consensus'],.77)
        votes['B']['start']={**vote,'distance_to_endpoint':16}
        self.assertEqual(consensus_edges([edge],votes,mode='require'),[])
        self.assertAlmostEqual(consensus_edges([edge],votes,bonus=.05)[0]['cosine_consensus'],.72)


if __name__=='__main__':unittest.main()
