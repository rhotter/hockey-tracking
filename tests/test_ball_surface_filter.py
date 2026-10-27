import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]/'experiments/ball-temporal-20261026'
spec = importlib.util.spec_from_file_location('surface_filter', HERE/'score_surface.py')
surface = importlib.util.module_from_spec(spec); spec.loader.exec_module(surface)


class SurfaceFilterTests(unittest.TestCase):
    def setUp(self):
        self.rink = {'active': True, 'height': 100,
                     'hull': [[20, 20], [80, 20], [80, 80], [20, 80]]}

    def test_inside_and_boundary(self):
        self.assertTrue(surface.inside_surface([40, 40], self.rink, 0))
        self.assertTrue(surface.inside_surface([20, 40], self.rink, 0))
        self.assertFalse(surface.inside_surface([19, 40], self.rink, 0))

    def test_margin_is_distance_to_polygon(self):
        self.assertTrue(surface.inside_surface([18, 40], self.rink, .025))
        self.assertFalse(surface.inside_surface([17, 40], self.rink, .025))
        self.assertFalse(surface.inside_surface([18, 18], self.rink, .025))

    def test_polygon_order_does_not_change_filter(self):
        reversed_rink = {**self.rink, 'hull': self.rink['hull'][::-1]}
        for point in [[40, 40], [18, 40], [17, 40], [18, 18]]:
            self.assertEqual(surface.inside_surface(point, self.rink, .025),
                             surface.inside_surface(point, reversed_rink, .025))

    def test_unusable_mask_does_not_claim_location_is_impossible(self):
        self.assertTrue(surface.inside_surface([1, 1], {**self.rink, 'active': False}, 0))


if __name__ == '__main__': unittest.main()
