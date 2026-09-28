from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from metarigor.local_run import RunFolder, RunManifest, RunStatus, StageIssue, StageOutcome, StageRunner, StageStatus
from .models import canonical_json
from .publication_coverage import compile_coverage, decide_headlines, evaluate_profile_compliance
from .publication_models import CoverageManifest, JournalPresentationProfile, ManuscriptDeliveryV2, ManuscriptFactPackageV2, ManuscriptRunRequestV2, ManuscriptRunResultV2, ProfileComplianceReport, PublicationDeliveryFile, PublicationVerificationReport
from .publication_rendering import render_publication
from .publication_verification import verify_publication
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True, slots=True)
class LocalPublicationResult:
    run_id: str
    package_id: str | None
    profile_id: str | None
    status: RunStatus
    run_root: Path
    delivery_path: str | None
    delivery_sha256: str | None


class LocalManuscriptV2Pipeline:
    """Manuscript V2: one sealed Fact Package and one Profile map to one non-overwritable Run."""

    def __init__(
        self,
        *,
        folder: RunFolder,
        manifest: RunManifest,
        request: ManuscriptRunRequestV2,
        agent_runtime=None,
        model_call_limiter=None,
    ) -> None:
        self.folder = folder
        self.manifest = manifest
        self.request = request
        self.agent_runtime = agent_runtime
        self.model_call_limiter = model_call_limiter
        self.stages = StageRunner(run_id=request.run_id, folder=folder, manifest=manifest)

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        request: ManuscriptRunRequestV2,
        agent_runtime=None,
        model_call_limiter=None,
    ) -> LocalManuscriptV2Pipeline:
        if request.writer_mode == "LLM" and agent_runtime is None:
            raise ValueError("LLM writer requires an explicit model runtime")
        folder = RunFolder.create(root.expanduser().resolve() / request.run_id)
        request_file = folder.write_immutable("inputs/request.json", canonical_json(request))
        manifest = RunManifest(folder.manifest_path)
        manifest.initialize()
        manifest.create_run(
            run_id=request.run_id,
            specialist="manuscript",
            input_path=request_file.path,
            input_sha256=request_file.sha256,
            started_at=_now(),
        )
        return cls(
            folder=folder,
            manifest=manifest,
            request=request,
            agent_runtime=agent_runtime,
            model_call_limiter=model_call_limiter
            if model_call_limiter is not None
            else asyncio.Semaphore(5),
        )

    async def run(self) -> LocalPublicationResult:
        self.manifest.set_run_status(self.request.run_id, RunStatus.RUNNING)
        try:
            return await self._run()
        except BaseException:
            try:
                self.manifest.set_run_status(
                    self.request.run_id, RunStatus.FAILED, finished_at=_now()
                )
            except ValueError:
                pass
            raise

    async def _run(self) -> LocalPublicationResult:
        imported = await self._import_inputs()
        if imported is None:
            return self._finish_failed(None, None)
        package, profile, profile_sha256 = imported
        compiled = await self._compile(package, profile, profile_sha256)
        if not compiled:
            return self._finish_failed(package.package_id, profile.profile_id)
        rendered = await self._render(package, profile, profile_sha256)
        if rendered is None:
            return self._finish_failed(package.package_id, profile.profile_id)
        compliance, coverage, rendered_paths = rendered
        verification = await self._verify(package, profile, coverage, compliance, rendered_paths)
        if verification is None:
            return self._finish_failed(package.package_id, profile.profile_id)
        if self.request.writer_mode == "LLM":
            from .publication_writing import review_writing

            reviewer = await review_writing(
                self,
                package,
                self.folder.read_verified(
                    "delivery/manuscript.md", rendered_paths["delivery/manuscript.md"]
                ).decode(),
                verification,
            )
            rendered_paths[reviewer.path] = reviewer.sha256
        delivery = await self._seal(
            package, profile, profile_sha256, compliance, verification, rendered_paths
        )
        if delivery is None:
            return self._finish_failed(package.package_id, profile.profile_id)
        delivery_model, delivery_path, delivery_sha256 = delivery
        has_issues = (
            bool(package.issues)
            or compliance.status != "PROFILE_COMPLETE"
            or verification.verdict != "PASSED"
            or delivery_model.readiness != "READY_FOR_HUMAN_FINALIZATION"
            or self.manifest.issue_count(self.request.run_id) > 0
        )
        status = RunStatus.COMPLETED_WITH_ISSUES if has_issues else RunStatus.SUCCEEDED
        result = ManuscriptRunResultV2(
            schema_version="manuscript-run-result.v2",
            run_id=self.request.run_id,
            package_id=package.package_id,
            profile_id=profile.profile_id,
            status=status.value,
            issue_count=self.manifest.issue_count(self.request.run_id),
            delivery_path=delivery_path,
            delivery_sha256=delivery_sha256,
        )
        final = self.folder.write_immutable("final.json", canonical_json(result))
        self.manifest.set_run_status(
            self.request.run_id,
            status,
            finished_at=_now(),
            final_output_path=final.path,
            final_output_sha256=final.sha256,
        )
        return LocalPublicationResult(
            run_id=self.request.run_id,
            package_id=package.package_id,
            profile_id=profile.profile_id,
            status=status,
            run_root=self.folder.root,
            delivery_path=delivery_path,
            delivery_sha256=delivery_sha256,
        )

    async def _import_inputs(
        self,
    ) -> tuple[ManuscriptFactPackageV2, JournalPresentationProfile, str] | None:
        package_path = _resolve_input(self.request.fact_package.input_path)
        profile_path = _resolve_input(self.request.journal_profile.input_path)

        def executor(_: dict) -> StageOutcome:
            package_bytes = package_path.read_bytes()
            profile_bytes = profile_path.read_bytes()
            if _sha256(package_bytes) != self.request.fact_package.expected_sha256:
                raise ValueError("Manuscript Fact Package V2 SHA-256 mismatch")
            if _sha256(profile_bytes) != self.request.journal_profile.expected_sha256:
                raise ValueError("Journal Profile SHA-256 mismatch")
            package = ManuscriptFactPackageV2.model_validate_json(package_bytes)
            profile = JournalPresentationProfile.model_validate_json(profile_bytes)
            if profile.profile_id != self.request.profile_id:
                raise ValueError("Journal Profile id does not match the Run request")
            package_copy = self.folder.write_immutable("inputs/fact-package.json", package_bytes)
            profile_copy = self.folder.write_immutable("inputs/journal-profile.json", profile_bytes)
            copied = [
                {"path": package_copy.path, "sha256": package_copy.sha256},
                {"path": profile_copy.path, "sha256": profile_copy.sha256},
            ]
            copied_by_sha = {package_copy.sha256: package_copy.path}
            source_content_by_sha = {package_copy.sha256: package_bytes}
            if self.request.writer_mode == "LLM":
                from .publication_proxy import build_proxy

                if package.fact_scope != "REFERENCE_UPSTREAM_PROXY":
                    raise ValueError("LLM writer requires a reference-upstream proxy package")
                audit_bytes = {}
                for name, bound in (
                    ("source-reference-fact-package.json", self.request.reference_package),
                    ("input-curation-audit.json", self.request.input_curation),
                ):
                    content = _resolve_input(bound.input_path).read_bytes()
                    if _sha256(content) != bound.expected_sha256:
                        raise ValueError(f"LLM input audit SHA-256 mismatch: {name}")
                    item = self.folder.write_immutable(f"inputs/{name}", content)
                    copied.append({"path": item.path, "sha256": item.sha256})
                    audit_bytes[name] = content
                source = ManuscriptFactPackageV2.model_validate_json(
                    audit_bytes["source-reference-fact-package.json"]
                )
                expected_package, expected_audit = build_proxy(source)
                expected_audit["source_file_sha256"] = _sha256(
                    audit_bytes["source-reference-fact-package.json"]
                )
                if (
                    package != expected_package
                    or json.loads(audit_bytes["input-curation-audit.json"]) != expected_audit
                ):
                    raise ValueError(
                        "Reference proxy or curation audit does not match its sealed source"
                    )
            for binding in (package.source_package_v1, *package.research_artifacts):
                source = _resolve_input(binding.input_path)
                content = source.read_bytes()
                if _sha256(content) != binding.expected_sha256:
                    raise ValueError(f"Research artifact SHA-256 mismatch: {binding.object_id}")
                suffix = source.suffix or ".bin"
                component = RunFolder.item_component(binding.object_id)
                target = f"inputs/research-artifacts/{component}{suffix}"
                item = self.folder.write_immutable(target, content)
                copied.append(
                    {"object_id": binding.object_id, "path": item.path, "sha256": item.sha256}
                )
                copied_by_sha[item.sha256] = item.path
                source_content_by_sha[item.sha256] = content
            profile_root = profile_path.parent.resolve()
            for binding in profile.source_bindings:
                source = (profile_root / binding.snapshot_path).resolve()
                if profile_root not in source.parents:
                    raise ValueError("Journal Profile snapshot escapes its profile root")
                content = source.read_bytes()
                if _sha256(content) != binding.snapshot_sha256:
                    raise ValueError(f"Journal Profile snapshot changed: {binding.source_id}")
                target = (
                    f"inputs/profile-sources/{RunFolder.item_component(binding.source_id)}.json"
                )
                item = self.folder.write_immutable(target, content)
                copied.append(
                    {"source_id": binding.source_id, "path": item.path, "sha256": item.sha256}
                )
            anchor_index = []
            for fact in package.facts:
                terminal_by_anchor = {
                    binding.source_anchor_id: binding for binding in fact.terminal_source_bindings
                }
                for anchor in fact.source_anchors:
                    content = source_content_by_sha.get(anchor.document_sha256)
                    if content is None:
                        raise ValueError(
                            f"Source Anchor document is not imported: {anchor.anchor_id}"
                        )
                    try:
                        text = content.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError(
                            f"Source Anchor document is not UTF-8 text: {anchor.anchor_id}"
                        ) from error
                    if text[anchor.start_offset : anchor.end_offset] != anchor.quote:
                        raise ValueError(f"Source Anchor quote mismatch: {anchor.anchor_id}")
                    terminal = terminal_by_anchor.get(anchor.anchor_id)
                    if terminal is not None:
                        terminal_content = source_content_by_sha.get(
                            terminal.terminal_document_sha256
                        )
                        if terminal_content is None:
                            raise ValueError(
                                "Derived Source Anchor terminal document is not imported: "
                                f"{anchor.anchor_id}"
                            )
                        view_content = source_content_by_sha.get(terminal.terminal_view_sha256)
                        if view_content is None:
                            raise ValueError(
                                "Derived Source Anchor Extraction View is not imported: "
                                f"{anchor.anchor_id}"
                            )
                        try:
                            view_text = view_content.decode("utf-8")
                        except UnicodeDecodeError as error:
                            raise ValueError(
                                f"Extraction View is not UTF-8 text: {anchor.anchor_id}"
                            ) from error
                        view_start = terminal.terminal_view_start_offset
                        view_end = terminal.terminal_view_end_offset
                        if view_text[view_start:view_end] != anchor.quote:
                            raise ValueError(
                                "Derived Source Anchor quote is absent at its bound "
                                f"Extraction View span: {anchor.anchor_id}"
                            )
                    anchor_index.append(
                        {
                            "fact_id": fact.fact_id,
                            "anchor_id": anchor.anchor_id,
                            "document_sha256": anchor.document_sha256,
                            "run_path": copied_by_sha[anchor.document_sha256],
                            "start_offset": anchor.start_offset,
                            "end_offset": anchor.end_offset,
                            "quote_sha256": anchor.quote_sha256,
                            "terminal_document_sha256": (
                                terminal.terminal_document_sha256 if terminal else None
                            ),
                            "terminal_view_sha256": (
                                terminal.terminal_view_sha256 if terminal else None
                            ),
                            "terminal_run_path": (
                                copied_by_sha[terminal.terminal_document_sha256]
                                if terminal is not None
                                else None
                            ),
                            "terminal_view_run_path": (
                                copied_by_sha[terminal.terminal_view_sha256]
                                if terminal is not None
                                else None
                            ),
                            "terminal_view_start_offset": (
                                terminal.terminal_view_start_offset if terminal else None
                            ),
                            "terminal_view_end_offset": (
                                terminal.terminal_view_end_offset if terminal else None
                            ),
                        }
                    )
            anchor_file = self.folder.write_immutable(
                "inputs/source-anchor-index.json",
                canonical_json(
                    {
                        "schema_version": "manuscript-source-anchor-index.v1",
                        "anchors": anchor_index,
                    }
                ),
            )
            copied.append({"path": anchor_file.path, "sha256": anchor_file.sha256})
            stage_issues = tuple(
                StageIssue(category=item.category, severity=item.severity, detail=item.detail)
                for item in package.issues
            )
            return StageOutcome(
                payload={
                    "schema_version": "manuscript-publication-import.v1",
                    "package_id": package.package_id,
                    "package_sha256": package_copy.sha256,
                    "profile_id": profile.profile_id,
                    "profile_sha256": profile_copy.sha256,
                    "copied_inputs": copied,
                },
                status=(
                    StageStatus.COMPLETED_WITH_ISSUES if stage_issues else StageStatus.SUCCEEDED
                ),
                issues=stage_issues,
            )

        result = await self.stages.execute(
            stage="publication_input_import",
            item_id="package-profile",
            payload={
                "fact_package_path": str(package_path),
                "fact_package_sha256": self.request.fact_package.expected_sha256,
                "profile_path": str(profile_path),
                "profile_sha256": self.request.journal_profile.expected_sha256,
                **(
                    {
                        "writer_mode": "LLM",
                        "reference_package": self.request.reference_package.model_dump(mode="json"),
                        "input_curation": self.request.input_curation.model_dump(mode="json"),
                    }
                    if self.request.writer_mode == "LLM"
                    else {}
                ),
            },
            executor=executor,
        )
        if result.status is StageStatus.FAILED:
            return None
        package = ManuscriptFactPackageV2.model_validate_json(
            self.folder.resolve("inputs/fact-package.json").read_bytes()
        )
        profile_bytes = self.folder.resolve("inputs/journal-profile.json").read_bytes()
        profile = JournalPresentationProfile.model_validate_json(profile_bytes)
        return package, profile, _sha256(profile_bytes)

    async def _compile(self, package, profile, profile_sha256) -> CoverageManifest | None:
        def executor(_: dict) -> StageOutcome:
            decisions = [asdict(item) for item in decide_headlines(package)]
            projection = {
                "schema_version": "manuscript-writer-projection.v2",
                "package_id": package.package_id,
                "profile_id": profile.profile_id,
                "facts": [
                    {
                        "fact_id": fact.fact_id,
                        "kind": fact.kind,
                        "status": fact.status,
                        "statement_en": fact.statement_en,
                        "required_for": list(fact.required_for),
                        "allowed_placements": list(fact.allowed_placements),
                        "citation_keys": list(fact.citation_keys),
                        "issue_refs": list(fact.issue_refs),
                    }
                    for fact in package.facts
                ],
                "headline_decisions": decisions,
            }
            projection_file = self.folder.write_immutable(
                "inputs/writer-projection.json", canonical_json(projection)
            )
            return StageOutcome(
                payload={
                    "schema_version": "manuscript-publication-plan.v1",
                    "writer_projection_path": projection_file.path,
                    "writer_projection_sha256": projection_file.sha256,
                    "headline_decisions": decisions,
                }
            )

        result = await self.stages.execute(
            stage="publication_profile_compile",
            item_id="manuscript",
            payload={
                "package_sha256": package.canonical_sha256,
                "profile_sha256": profile_sha256,
            },
            executor=executor,
        )
        if result.status is StageStatus.FAILED:
            return None
        return True

    async def _render(self, package, profile, profile_sha256):
        evidence = None
        assembly = None
        if self.request.writer_mode == "LLM":
            from .publication_writing import assemble_writing, write_manuscript

            evidence = await write_manuscript(self, package, profile)
            if evidence is None:
                return None
            assembly = assemble_writing(package, profile, evidence)

        def executor(_: dict) -> StageOutcome:
            rendered = (
                assembly.rendered if assembly is not None else render_publication(package, profile)
            )
            files = {}
            payloads = {
                "delivery/manuscript.md": rendered.manuscript.encode("utf-8"),
                "delivery/tables.md": rendered.tables.encode("utf-8"),
                "delivery/supplement.md": rendered.supplement.encode("utf-8"),
                "delivery/references.md": rendered.references.encode("utf-8"),
                "delivery/figures/prisma-flow.md": rendered.prisma_flow.encode("utf-8"),
                "delivery/figures/forest-plot.svg": rendered.forest_svg,
                "delivery/claim-lineage.jsonl": rendered.claim_lineage_jsonl,
            }
            if assembly is not None:
                payloads["delivery/writing-evidence.json"] = canonical_json(evidence)
                payloads["delivery/claim-occurrences.json"] = canonical_json(assembly.occurrences)
            for path, content in payloads.items():
                item = self.folder.write_immutable(path, content)
                files[path] = item.sha256
            coverage = compile_coverage(
                package,
                profile,
                manuscript=rendered.manuscript,
                tables=rendered.tables,
                supplement=rendered.supplement,
                references=rendered.references,
                prisma_flow=rendered.prisma_flow,
                forest_svg=rendered.forest_svg,
                section_ids=rendered.section_ids,
                front_matter_modules=rendered.front_matter_modules,
                main_object_ids=rendered.main_object_ids,
                supplement_object_ids=rendered.supplement_object_ids,
                supplement_modules=rendered.supplement_modules,
                fact_presence=assembly.presence if assembly is not None else None,
                semantic_unverified=assembly.semantic_unverified if assembly is not None else None,
            )
            coverage_file = self.folder.write_immutable(
                "delivery/prisma-placement.json", canonical_json(coverage)
            )
            files[coverage_file.path] = coverage_file.sha256
            compliance = evaluate_profile_compliance(
                package=package,
                profile=profile,
                profile_sha256=profile_sha256,
                title=rendered.title,
                abstract=rendered.abstract,
                manuscript=rendered.manuscript,
                tables=rendered.tables,
                supplement=rendered.supplement,
                references=rendered.references,
                coverage=coverage,
                section_ids=rendered.section_ids,
                front_matter_modules=rendered.front_matter_modules,
                main_object_ids=rendered.main_object_ids,
                supplement_object_ids=rendered.supplement_object_ids,
                supplement_modules=rendered.supplement_modules,
            )
            item = self.folder.write_immutable(
                "delivery/profile-compliance.json", canonical_json(compliance)
            )
            files[item.path] = item.sha256
            return StageOutcome(
                payload={
                    "schema_version": "manuscript-publication-render.v1",
                    "files": files,
                    "coverage": coverage.model_dump(mode="json"),
                    "compliance": compliance.model_dump(mode="json"),
                    "main_table_figure_count": rendered.main_table_figure_count,
                },
                status=(
                    StageStatus.COMPLETED_WITH_ISSUES
                    if compliance.status != "PROFILE_COMPLETE"
                    else StageStatus.SUCCEEDED
                ),
                issues=(
                    tuple(
                        StageIssue(category=i.category, severity=i.severity, detail=i.detail)
                        for i in assembly.issues
                    )
                    if assembly
                    else ()
                )
                + (
                    (
                        StageIssue(
                            category="profile_not_ready_for_human_finalization",
                            severity="WARNING",
                            detail=f"Profile compliance status is {compliance.status}.",
                        ),
                    )
                    if compliance.status != "PROFILE_COMPLETE"
                    else ()
                ),
            )

        result = await self.stages.execute(
            stage="publication_render",
            item_id="manuscript",
            payload={
                "package_sha256": package.canonical_sha256,
                "profile_sha256": profile_sha256,
                **(
                    {"writer_mode": "LLM", "writing_evidence": evidence}
                    if evidence is not None
                    else {}
                ),
            },
            executor=executor,
        )
        if result.status is StageStatus.FAILED:
            return None
        payload = self._read(result)
        return (
            ProfileComplianceReport.model_validate(payload["compliance"]),
            CoverageManifest.model_validate(payload["coverage"]),
            payload["files"],
        )

    async def _verify(self, package, profile, coverage, compliance, rendered_paths):
        def text(path: str) -> str:
            return self.folder.read_verified(path, rendered_paths[path]).decode("utf-8")

        def executor(_: dict) -> StageOutcome:
            report = verify_publication(
                package=package,
                profile=profile,
                manuscript=text("delivery/manuscript.md"),
                tables=text("delivery/tables.md"),
                supplement=text("delivery/supplement.md"),
                references=text("delivery/references.md"),
                prisma_flow=text("delivery/figures/prisma-flow.md"),
                forest_svg=self.folder.read_verified(
                    "delivery/figures/forest-plot.svg",
                    rendered_paths["delivery/figures/forest-plot.svg"],
                ),
                claim_lineage_jsonl=self.folder.read_verified(
                    "delivery/claim-lineage.jsonl",
                    rendered_paths["delivery/claim-lineage.jsonl"],
                ),
                coverage=coverage,
                compliance=compliance,
                writing_evidence=(
                    json.loads(text("delivery/writing-evidence.json"))
                    if self.request.writer_mode == "LLM"
                    else None
                ),
            )
            item = self.folder.write_immutable(
                "delivery/verifier-report.json", canonical_json(report)
            )
            issues = tuple(
                StageIssue(category=value.category, severity=value.severity, detail=value.detail)
                for value in report.issues
            )
            return StageOutcome(
                payload={
                    "schema_version": "manuscript-publication-verification-stage.v1",
                    "report": report.model_dump(mode="json"),
                    "path": item.path,
                    "sha256": item.sha256,
                },
                status=(StageStatus.COMPLETED_WITH_ISSUES if issues else StageStatus.SUCCEEDED),
                issues=issues,
            )

        result = await self.stages.execute(
            stage="publication_verify",
            item_id="manuscript",
            payload={
                "package_sha256": package.canonical_sha256,
                "profile_sha256": profile.canonical_sha256,
                "coverage_sha256": coverage.canonical_sha256,
                "compliance_sha256": compliance.canonical_sha256,
                "rendered_files": rendered_paths,
            },
            executor=executor,
        )
        if result.status is StageStatus.FAILED:
            return None
        return PublicationVerificationReport.model_validate(self._read(result)["report"])

    async def _seal(
        self, package, profile, profile_sha256, compliance, verification, rendered_paths
    ):
        def executor(_: dict) -> StageOutcome:
            files = dict(rendered_paths)
            for path in (
                "delivery/prisma-placement.json",
                "delivery/verifier-report.json",
            ):
                content = self.folder.resolve(path).read_bytes()
                files[path] = _sha256(content)
            role_by_path = {
                "delivery/manuscript.md": "MANUSCRIPT",
                "delivery/tables.md": "TABLES",
                "delivery/supplement.md": "SUPPLEMENT",
                "delivery/references.md": "REFERENCES",
                "delivery/figures/prisma-flow.md": "PRISMA_FLOW",
                "delivery/figures/forest-plot.svg": "FOREST_PLOT",
                "delivery/claim-lineage.jsonl": "CLAIM_LINEAGE",
                "delivery/prisma-placement.json": "PRISMA_PLACEMENT",
                "delivery/profile-compliance.json": "PROFILE_COMPLIANCE",
                "delivery/verifier-report.json": "VERIFIER",
                "delivery/writing-evidence.json": "WRITING_EVIDENCE",
                "delivery/claim-occurrences.json": "CLAIM_OCCURRENCES",
                "delivery/reviewer-report.json": "REVIEWER",
            }
            decisions = decide_headlines(package)
            readiness = (
                "AUDIT_ONLY"
                if any(item.eligibility == "AUDIT_ONLY" for item in decisions)
                else "READY_FOR_HUMAN_FINALIZATION"
                if compliance.status == "PROFILE_COMPLETE"
                and verification.verdict == "PASSED"
                and self.manifest.issue_count(self.request.run_id) == 0
                else "NOT_READY_FOR_HUMAN_FINALIZATION"
            )
            delivery = ManuscriptDeliveryV2(
                schema_version="manuscript-delivery.v2",
                run_id=self.request.run_id,
                package_id=package.package_id,
                profile_id=profile.profile_id,
                profile_sha256=profile_sha256,
                files=tuple(
                    PublicationDeliveryFile(role=role_by_path[path], path=path, sha256=digest)
                    for path, digest in sorted(files.items())
                ),
                issue_count=self.manifest.issue_count(self.request.run_id),
                readiness=readiness,
            )
            item = self.folder.write_immutable("delivery/delivery.json", canonical_json(delivery))
            return StageOutcome(
                payload={
                    "schema_version": "manuscript-publication-seal.v1",
                    "delivery": delivery.model_dump(mode="json"),
                    "delivery_path": item.path,
                    "delivery_sha256": item.sha256,
                }
            )

        result = await self.stages.execute(
            stage="publication_seal",
            item_id="manuscript",
            payload={
                "profile_sha256": profile_sha256,
                "compliance_sha256": compliance.canonical_sha256,
                "verification_sha256": verification.canonical_sha256,
                "rendered_files": rendered_paths,
            },
            executor=executor,
        )
        if result.status is StageStatus.FAILED:
            return None
        payload = self._read(result)
        return (
            ManuscriptDeliveryV2.model_validate(payload["delivery"]),
            payload["delivery_path"],
            payload["delivery_sha256"],
        )

    def _finish_failed(self, package_id: str | None, profile_id: str | None):
        result = ManuscriptRunResultV2(
            schema_version="manuscript-run-result.v2",
            run_id=self.request.run_id,
            package_id=package_id or "unknown",
            profile_id=profile_id or "unknown",
            status="FAILED",
            issue_count=self.manifest.issue_count(self.request.run_id),
            delivery_path=None,
            delivery_sha256=None,
        )
        final = self.folder.write_immutable("final.json", canonical_json(result))
        self.manifest.set_run_status(
            self.request.run_id,
            RunStatus.FAILED,
            finished_at=_now(),
            final_output_path=final.path,
            final_output_sha256=final.sha256,
        )
        return LocalPublicationResult(
            run_id=self.request.run_id,
            package_id=package_id,
            profile_id=profile_id,
            status=RunStatus.FAILED,
            run_root=self.folder.root,
            delivery_path=None,
            delivery_sha256=None,
        )

    def _read(self, result) -> dict:
        if result.output_path is None or result.output_sha256 is None:
            raise ValueError("Stage Result has no readable output")
        return json.loads(self.folder.read_verified(result.output_path, result.output_sha256))


def _resolve_input(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()

