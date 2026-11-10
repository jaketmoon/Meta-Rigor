"""Preserve the repository-relative paths used by frozen experiment manifests."""

import json
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithResearchData(build_py):
    def run(self):
        super().run()
        source = Path(__file__).resolve().parent
        target = Path(self.build_lib) / "metarigor" / "_research"
        manifest = "inputs/checksums.json"
        with (source / manifest).open(encoding="utf-8") as stream:
            assets = set(json.load(stream))
        assets.add(manifest)
        assets.update(
            f"runtime/openhands-cli/{name}"
            for name in ("Dockerfile", "model_proxy.py", "restricted_sitecustomize.py")
        )
        for relative_path in sorted(assets):
            destination = target / relative_path
            self.mkpath(str(destination.parent))
            self.copy_file(str(source / relative_path), str(destination))


setup(cmdclass={"build_py": BuildWithResearchData})
