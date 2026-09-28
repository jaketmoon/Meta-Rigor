from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ImmutableFile:
    path: str
    sha256: str
    byte_size: int


class RunFolder:
    """Self-contained Run directory; files other than the manifest use relative references."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @classmethod
    def create(cls, root: Path) -> RunFolder:
        resolved = root.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=False)
        return cls(resolved)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.sqlite"

    def resolve(self, relative_path: str) -> Path:
        candidate = PurePosixPath(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise ValueError("Run-relative path must stay inside the Run Folder")
        resolved = self.root.joinpath(*candidate.parts).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("Run-relative path escapes the Run Folder")
        return resolved

    def write_immutable(self, relative_path: str, content: bytes) -> ImmutableFile:
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(content).hexdigest()
        if target.exists():
            existing = target.read_bytes()
            if hashlib.sha256(existing).hexdigest() != digest:
                raise FileExistsError(
                    f"immutable Run file already exists with other content: {relative_path}"
                )
            return ImmutableFile(relative_path, digest, len(existing))
        descriptor, temporary_name = tempfile.mkstemp(prefix=".stage-", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                existing = target.read_bytes()
                if hashlib.sha256(existing).hexdigest() != digest:
                    raise
            directory_descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        return ImmutableFile(relative_path, digest, len(content))

    def read_verified(self, relative_path: str, expected_sha256: str) -> bytes:
        content = self.resolve(relative_path).read_bytes()
        observed = hashlib.sha256(content).hexdigest()
        if observed != expected_sha256:
            raise ValueError(f"Run file SHA-256 mismatch: {relative_path}")
        return content

    @staticmethod
    def item_component(item_id: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", item_id).strip("-._")[:48] or "item"
        suffix = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:12]
        return f"{slug}-{suffix}"

    @classmethod
    def stage_paths(cls, stage: str, item_id: str, input_sha256: str) -> tuple[str, str]:
        stage_component = cls.item_component(stage)
        item_component = cls.item_component(item_id)
        base = f"stages/{stage_component}/{item_component}/{input_sha256[:16]}"
        return f"{base}/input.json", f"{base}/output.json"

