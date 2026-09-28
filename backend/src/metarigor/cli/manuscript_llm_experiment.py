from __future__ import annotations

"""One-shot MS V2 LLM experiment: new package, fresh candidate, original seven-dimensional Judge, and all failures summarized."""


import hashlib
from metarigor.manuscript.models import canonical_json
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else canonical_json(value))

