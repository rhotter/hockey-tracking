# Project scope

- NBHL ball hockey only; do not introduce ice- or roller-hockey benchmark claims.
- Preserve `experiments/reference/` and its frozen hashes. Put new experiments elsewhere.
- Keep sparse pair checks distinct from continuous/full-game identity accuracy.
- Automatic track IDs are not player identities, shifts, time on floor or metric distance.
- Never commit credentials, private hosting metadata, raw video, crops, model weights or downloaded vendor code.
- Run `python scripts/verify_results.py`, `python scripts/verify_release.py`, and `python -m unittest discover -s tests -v` after relevant changes.
- Update `release-manifest.json` intentionally when changing published artifacts; never use changed reference labels to claim a fresh independent test.
