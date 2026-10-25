import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/overhead-20261025/scripts'))
from geometry import project,usable,floor_point,footpoint,segment_path,visible_polygon

class OverheadGeometryTests(unittest.TestCase):
    def setUp(self):self.H=np.array([[7.637/1920,0,-4.005],[0,2/1080,-1],[0,0,1]])
    def test_roundtrip_with_perspective(self):
        H=self.H@np.array([[1,.02,20],[.01,1,-10],[.00002,-.00003,1]])
        p=[[100,100],[800,400],[1600,950]]
        np.testing.assert_allclose(project(np.linalg.inv(H),project(H,p)),p,atol=1e-8)
    def test_invalid_and_horizon(self):
        self.assertFalse(usable(None));self.assertFalse(usable(np.zeros((3,3))))
        G=np.array([[100,0,0],[0,100,0],[1,0,0.1]])
        self.assertFalse(usable(np.linalg.inv(G)))
        self.assertIsNone(floor_point(self.H,[-50,300]))
    def test_known_floor_location(self):
        self.assertTrue(usable(self.H));np.testing.assert_allclose(floor_point(self.H,[960,540]),[.5,.5])
    def test_camera_footprint_is_clipped_to_rink(self):
        polygon=visible_polygon(self.H)
        self.assertEqual(len(polygon),4)
        np.testing.assert_allclose(polygon,[[0,0],[1,0],[1,1],[0,1]],atol=1e-8)
        H=self.H@np.array([[1,0,960],[0,1,0],[0,0,1]])
        p=np.array(visible_polygon(H));self.assertAlmostEqual(p[:,0].min(),.5);self.assertAlmostEqual(p[:,0].max(),1.)
    def test_feet_require_visible_evidence(self):
        good={'ankles':[[130,290],[160,293]],'scores':[.9,.85]}
        self.assertIsNotNone(footpoint(good,[100,100,180,300]))
        self.assertIsNone(footpoint(good,[100,100,180,1080]))
        self.assertIsNone(footpoint(dict(good,scores=[.2,.9]),[100,100,180,300]))
        self.assertIsNone(footpoint(dict(good,scores=[float('nan'),.9]),[100,100,180,300]))
        self.assertIsNone(footpoint(dict(good,ankles=[[130,120],[160,121]]),[100,100,180,300]))
    def test_no_paths_across_missing_frames_or_identity_changes(self):
        rows=[{'frame':f,'track_id':tid,'xy':xy}for f,tid,xy in [(0,'a',[.1,.2]),(1,'a',[.101,.201]),(3,'a',[.102,.202]),(4,'b',[.103,.203]),(5,'b',None),(6,'b',[.104,.204]),(7,'b',[.8,.8])]]
        out=segment_path(rows)
        self.assertEqual([r['segment']for r in out],[0,0,1,2,None,3,4])

if __name__=='__main__':unittest.main()
