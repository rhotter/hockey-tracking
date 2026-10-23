"""Synthetic API example; these numbers are not experimental evidence."""
import json

from hockey_tracking.reconnect import reconnect

tracks = {"clip": "synthetic", "frames": [
    {"frame": 0, "objects": [{"track_id": "17", "bbox": [10, 20, 30, 60]}]},
    {"frame": 45, "objects": [{"track_id": "42", "bbox": [35, 20, 55, 60]}]},
]}
evidence = {"17": {"start": 0, "end": 14}, "42": {"start": 45, "end": 59}}
candidates = [{"a": "17", "b": "42", "gap": 31,
               "distance_heights": 0.7, "support_a": 15, "support_b": 15,
               "cosine_mean": 0.8, "cosine_joint": 0.9}]

print(json.dumps(reconnect(tracks, evidence, candidates), indent=2))
