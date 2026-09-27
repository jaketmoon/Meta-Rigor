from __future__ import annotations

"""MS V2 LLM 一次性实验入口：新包、fresh candidate、原七维 Judge、失败全量汇总。"""


import hashlib
from metarigor.manuscript.models import canonical_json
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else canonical_json(value))


