"""Safety gates around upstream components; no model downloads in unit tests."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'experiments/roboflow-20261025/scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('roboflow_adapter',SCRIPTS/'nbhl_roboflow.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)

class RoboflowAdapterTests(unittest.TestCase):
    def reading(self,values,frames=None,identity='yes'):
        return adapter.three_readings({'identity_consistent':identity,'readings':[{'image':i+1,'number':n} for i,n in enumerate(values)]},
                                      [{'frame':f} for f in (frames if frames is not None else range(len(values)))])

    def test_unknown_reading_breaks_run(self):
        self.assertIsNone(self.reading([12,None,12,12]))
        self.assertEqual(self.reading([None,12,12,12]),12)

    def test_image_order_is_not_time_order(self):
        self.assertIsNone(self.reading([12,12,12,None],[0,2,3,1]))
        self.assertEqual(self.reading([12,None,12,12],[2,3,0,1]),12)

    def test_ambiguous_identity_and_conflicting_runs_abstain(self):
        self.assertIsNone(self.reading([12,12,12],identity='uncertain'))
        self.assertIsNone(self.reading([12,12,12,14,14,14]))

    def test_malformed_or_duplicate_readings_abstain(self):
        self.assertIsNone(adapter.three_readings({'identity_consistent':'yes','readings':None},[{'frame':0}]))
        self.assertIsNone(adapter.three_readings({'identity_consistent':'yes','readings':[{'image':1,'number':12}]*3},[{'frame':f} for f in range(3)]))
        self.assertIsNone(self.reading([True,True,True]))

    def test_team_distance_and_margin_both_required(self):
        accepted,_=adapter.team_gate([[1,3],[1,1.1],[3,10],[float('nan'),1]],radius=2)
        self.assertEqual(accepted.tolist(),[True,False,False,False])

if __name__=='__main__':unittest.main()
