"""Verify the public artifact manifest, reference source and release hygiene."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    patterns = [
        rb"hf_[A-Za-z0-9]{25,}",
        rb"gh[pousr]_[A-Za-z0-9]{25,}",
        rb"github_pat_[A-Za-z0-9_]{25,}",
        rb"sk-(?:proj-)?[A-Za-z0-9_-]{30,}",
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        rb"/" + rb"home/[^/\s]+/",
        rb"appg(?:prj|dep|ver)_[A-Za-z0-9]+",
    ]
    total = 0
    for name, expected in manifest["files"].items():
        path = ROOT / name
        assert not path.is_symlink(), name
        assert path.resolve().is_relative_to(ROOT), name
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected, name
        assert path.suffix.lower() not in {".mp4", ".mov", ".pt", ".pth", ".safetensors"}, name
        assert not path.name.startswith(".env"), name
        for pattern in patterns:
            assert not re.search(pattern, content), f"Release hygiene match in {name}"
        total += len(content)
    provenance = json.loads((ROOT / "experiments/provenance.json").read_text())
    for name, expected in provenance["reference_sha256"].items():
        path = ROOT / "experiments/reference" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    print(json.dumps({"files_verified": len(manifest["files"]), "bytes": total,
                      "reference_files_verified": len(provenance["reference_sha256"]),
                      "credential_and_private_metadata_patterns": 0}))


if __name__ == "__main__":
    main()
