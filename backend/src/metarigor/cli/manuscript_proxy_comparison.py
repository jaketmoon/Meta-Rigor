from __future__ import annotations

"""用户授权的 MS proxy 重评与两个单轮消融；复用 FULL 正文，另行封存结果。"""


from pathlib import Path
from metarigor.manuscript import ms_ablation as ablation
from metarigor.manuscript.models import canonical_json
from .manuscript_llm_experiment import digest, write
def copy_tree(source, target):
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Experiment source must not contain symlinks")
        if path.is_file():
            write(target / path.relative_to(source), path.read_bytes())


def initialize_ablation(root, condition, full, runtime):
    """仅替换旧初始化器的 deterministic FULL pin；复用原消融写作和装配行为。"""
    target = root / "conditions" / condition.lower()
    manifest = {
        "schema_version": "ms-proxy-ablation.v1",
        "condition": condition,
        "case_ids": list(full),
        "case_bindings": {},
    }
    for case_id in full:
        row, candidate = full[case_id]
        full_root = Path(row["run_root"])
        package_root = target / "packages" / case_id
        for source_name, target_name in (
            ("fact-package.json", "fact-package.json"),
            ("journal-profile.json", "journal-profile.json"),
            ("source-reference-fact-package.json", "source-reference-fact-package.json"),
            ("input-curation-audit.json", "input-curation-audit.json"),
            ("source-anchor-index.json", "source-anchor-index.json"),
        ):
            write(package_root / target_name, (full_root / "inputs" / source_name).read_bytes())
        for name in ("research-artifacts", "profile-sources"):
            copy_tree(full_root / "inputs" / name, package_root / name)
        projection = canonical_json(candidate.projection)
        if ablation._sha(projection) != candidate.projection_sha256:
            raise ValueError("Source writer projection bytes differ from canonical projection")
        write(package_root / "writer-projection.json", projection)
        source_surfaces = {}
        for role in ablation._SUPPORT_SURFACES:
            surface = candidate.surfaces[role]
            content = (full_root / surface.path).read_bytes()
            if ablation._sha(content) != surface.sha256:
                raise ValueError("FULL support surface changed")
            relative = f"source-full-surfaces/{Path(surface.path).name}"
            write(package_root / relative, content)
            source_surfaces[role] = {"path": relative, "sha256": surface.sha256}
        manuscript = candidate.surfaces["MANUSCRIPT"].content
        offset = manuscript.find(ablation._PROGRAM_SUPPORT_HEADING)
        if offset < 0:
            raise ValueError("FULL manuscript has no deterministic support tail")
        tail = manuscript[offset:].encode()
        relative = "source-full-surfaces/continuous-support-tail.md"
        write(package_root / relative, tail)
        source_surfaces["CONTINUOUS_SUPPORT_TAIL"] = {
            "path": relative,
            "sha256": ablation._sha(tail),
        }
        files = [
            {"path": str(p.relative_to(package_root)), "sha256": digest(p)}
            for p in sorted(package_root.rglob("*"))
            if p.is_file()
        ]
        inputs = {
            "schema_version": "ms-proxy-ablation-input.v1",
            "condition": condition,
            "case_id": case_id,
            "source_full_run_id": row["run_id"],
            "source_full_delivery_sha256": candidate.delivery_sha256,
            "package_sha256": candidate.fact_package_sha256,
            "profile_sha256": candidate.journal_profile_sha256,
            "projection_sha256": candidate.projection_sha256,
            "source_surfaces": source_surfaces,
            "writer_runtime": runtime.binding("worker"),
            "files": files,
            "references_policy": "CITATION_KEY_AND_DISPLAY_TEXT_ONLY",
            "value_slots_required": False,
        }
        write(package_root / "input-manifest.json", inputs)
        manifest["case_bindings"][case_id] = {
            "input_manifest_sha256": digest(package_root / "input-manifest.json"),
            "package_sha256": candidate.fact_package_sha256,
        }
    write(target / "manifest.json", manifest)
    return target, manifest


