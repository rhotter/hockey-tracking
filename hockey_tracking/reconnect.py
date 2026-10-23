"""The fixed joint-image 2B challenger, operating on precomputed candidates.

Inputs must already satisfy scene/candidate construction from the experiment.
This module does not detect players or generate image embeddings.
"""
import math

from .linking import global_links

SETTINGS = {
    "solver": "global_assignment",
    "pooling": "embedding",
    "cosine": 0.75,
    "margin": 0.0,
    "gap": 300,
    "gap_penalty": 0.001,
    "motion_base": 1.0,
    "motion_per_frame": 0.12,
    "min_support": 10,
}


def reconnect(result, evidence, candidates, *, fps=15):
    """Reconnect observed fragments using cached OSNet and joint-Qwen cosines.

    A candidate's ``cosine_joint`` is None/absent if it was not shortlisted.
    Such edges are excluded, matching the measured blend policy. Both cosine
    values must come from normalized features, not language-model confidence.
    The original solver's gap penalty assumes 15 fps; reject other rates.
    """
    if fps != 15:
        raise ValueError("This frozen configuration requires 15 fps.")
    scored = []
    for edge in candidates:
        joint = edge.get("cosine_joint")
        if joint is None:
            continue
        appearance = edge["cosine_mean"]
        if not all(math.isfinite(v) and -1.000001 <= v <= 1.000001
                   for v in (joint, appearance)):
            raise ValueError("Expected finite cosine similarities in [-1, 1].")
        scored.append(edge | {"cosine_embedding": 0.5 * (joint + appearance)})
    return global_links(result, evidence, scored, SETTINGS)
