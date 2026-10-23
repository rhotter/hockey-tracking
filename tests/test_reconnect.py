import copy
import importlib.util
from pathlib import Path
import unittest

from hockey_tracking.reconnect import reconnect, SETTINGS


class ReconnectionTests(unittest.TestCase):
    def setUp(self):
        self.tracks = {"frames": [
            {"frame": 0, "objects": [{"track_id": "a", "bbox": [0, 0, 10, 20]},
                                     {"track_id": "b", "bbox": [30, 0, 40, 20]}]},
            {"frame": 30, "objects": [{"track_id": "c", "bbox": [5, 0, 15, 20]},
                                      {"track_id": "d", "bbox": [35, 0, 45, 20]}]},
        ]}
        self.evidence = {k: {"start": 0, "end": 10} for k in "ab"}
        self.evidence.update({k: {"start": 30, "end": 40} for k in "cd"})

    def edge(self, a, b, score):
        return {"a": a, "b": b, "gap": 20, "support_a": 11, "support_b": 11,
                "distance_heights": 0, "cosine_mean": score, "cosine_joint": score}

    def test_global_competition_beats_greedy(self):
        edges = [self.edge("a", "c", .95), self.edge("a", "d", .9),
                 self.edge("b", "c", .9), self.edge("b", "d", .76)]
        out = reconnect(self.tracks, self.evidence, edges)
        self.assertEqual({(r["a"], r["b"]) for r in out["links"]}, {("a", "d"), ("b", "c")})

    def test_preserves_observations_and_does_not_mutate_input(self):
        original = copy.deepcopy(self.tracks)
        out = reconnect(self.tracks, self.evidence, [self.edge("a", "c", .9)])
        self.assertEqual(original, self.tracks)
        self.assertEqual([r["frame"] for r in out["frames"]], [0, 30])
        for before, after in zip(original["frames"], out["frames"]):
            self.assertEqual([o["bbox"] for o in before["objects"]],
                             [o["bbox"] for o in after["objects"]])
            self.assertEqual(len({o["track_id"] for o in after["objects"]}), len(after["objects"]))

    def test_rejects_ineligible_edges(self):
        for change in [{"gap": 301}, {"support_a": 9}, {"distance_heights": 100},
                       {"cosine_joint": None}, {"cosine_joint": .5}]:
            with self.subTest(change=change):
                edge = self.edge("a", "c", .9) | change
                self.assertEqual(reconnect(self.tracks, self.evidence, [edge])["links"], [])

    def test_never_joins_simultaneously_visible_fragments(self):
        out = reconnect(self.tracks, self.evidence, [self.edge("a", "b", .99)])
        self.assertEqual(out["links"], [])

    def test_rejects_unvalidated_frame_rate_and_nonfinite_score(self):
        with self.assertRaises(ValueError):
            reconnect(self.tracks, self.evidence, [], fps=30)
        with self.assertRaises(ValueError):
            reconnect(self.tracks, self.evidence, [self.edge("a", "c", float("nan"))])

    def test_matches_original_solver_on_candidate_graphs(self):
        import numpy as np
        path = Path(__file__).resolve().parents[1] / "experiments/reference/scripts/nbhl_tracking_global_links.py"
        spec = importlib.util.spec_from_file_location("original_links", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(17)
        for _ in range(30):
            edges = [self.edge(a, b, float(rng.uniform(.5, 1))) for a in "ab" for b in "cd"]
            expected = module.global_links(self.tracks, self.evidence,
                [r | {"cosine_embedding": .5*(r["cosine_mean"]+r["cosine_joint"])} for r in edges], SETTINGS)
            self.assertEqual(reconnect(self.tracks, self.evidence, edges), expected)


if __name__ == "__main__":
    unittest.main()
