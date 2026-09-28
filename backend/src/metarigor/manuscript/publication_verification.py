from __future__ import annotations

import json
import re
from .publication_coverage import decide_headlines
from .publication_models import CoverageManifest, JournalPresentationProfile, ManuscriptFactPackageV2, ProfileComplianceReport, PublicationVerificationIssue, PublicationVerificationReport
from .publication_rendering import render_publication
_CONTINUOUS_GENERIC_PROFILE_ID = "generic_prisma_2020_meta_analysis_v2"


def verify_publication(
    *,
    package: ManuscriptFactPackageV2,
    profile: JournalPresentationProfile,
    manuscript: str,
    tables: str,
    supplement: str,
    references: str,
    prisma_flow: str,
    forest_svg: bytes,
    claim_lineage_jsonl: bytes,
    coverage: CoverageManifest,
    compliance: ProfileComplianceReport,
    writing_evidence: dict | None = None,
) -> PublicationVerificationReport:
    if writing_evidence is not None:
        from .publication_writing import verify_writing

        if package.fact_scope != "REFERENCE_UPSTREAM_PROXY":
            raise ValueError("LLM verification requires its reference proxy contract")
        return verify_writing(
            package,
            profile,
            writing_evidence,
            {
                "manuscript": manuscript,
                "tables": tables,
                "supplement": supplement,
                "references": references,
                "prisma_flow": prisma_flow,
                "forest_svg": forest_svg,
                "claim_lineage_jsonl": claim_lineage_jsonl,
            },
            coverage,
            compliance,
        )
    issues: list[PublicationVerificationIssue] = []
    expected = render_publication(package, profile)
    noncanonical = tuple(
        name
        for name, observed, canonical in (
            ("manuscript", manuscript, expected.manuscript),
            ("tables", tables, expected.tables),
            ("supplement", supplement, expected.supplement),
            ("references", references, expected.references),
            ("prisma-flow", prisma_flow, expected.prisma_flow),
            ("forest", forest_svg, expected.forest_svg),
            ("claim-lineage", claim_lineage_jsonl, expected.claim_lineage_jsonl),
        )
        if observed != canonical
    )
    if noncanonical:
        issues.append(
            PublicationVerificationIssue(
                category="NONCANONICAL_DELIVERY_SURFACE",
                severity="ERROR",
                detail=(
                    "Deterministic delivery differs from the exact source-bound rendering in: "
                    f"{', '.join(noncanonical)}."
                ),
            )
        )
    forest = forest_svg.decode("utf-8")
    surface = "\n".join((manuscript, tables, supplement, references, prisma_flow, forest))
    if coverage.required_count != coverage.covered_count or any(
        item.status == "FAILED" for item in coverage.items
    ):
        issues.append(
            PublicationVerificationIssue(
                category="INCOMPLETE_REQUIRED_FACT_COVERAGE",
                severity="ERROR",
                detail=f"Coverage is {coverage.covered_count}/{coverage.required_count}.",
            )
        )
    if re.search(
        r"fact-[A-Za-z0-9]|pub-[A-Za-z0-9]|\[MISSING\]|\bTBD\b|\{\{|"
        r"[\u3400-\u9fff]|Fact Package|Run Folder|Manuscript Specialist|"
        r"Journal Presentation Profile|\bcompiler\b|\bsealed\b|\bV2 closure\b|"
        r"\bsource package\b|\bprogrammed consistency\b|\bCartesian closure\b",
        surface,
        re.IGNORECASE,
    ):
        issues.append(
            PublicationVerificationIssue(
                category="FORBIDDEN_PUBLICATION_SURFACE",
                severity="ERROR",
                detail=(
                    "A delivery surface contains an internal token, internal implementation "
                    "term, placeholder, template marker, or Chinese residual text."
                ),
            )
        )
    decisions = {item.analysis_id: item for item in decide_headlines(package)}
    section_text = {
        section.section_id: _section(manuscript, section.heading, profile.section_contract)
        for section in profile.section_contract
    }
    abstract = manuscript.split("## Abstract", 1)[1].split("## ", 1)[0]
    summary_table = tables.split("## Table 3. Summary of findings", 1)[1]
    synthesis_facts = [fact for fact in package.facts if fact.kind == "SYNTHESIS"]
    primary_analysis_id = next(
        (fact.analysis_id for fact in synthesis_facts if fact.is_primary),
        synthesis_facts[0].analysis_id,
    )
    for fact in package.facts:
        if fact.kind == "SYNTHESIS":
            if fact.validity == "NOT_REPORTED":
                label = _outcome_label(package, fact.outcome_id)
                table_row = f"| {fact.outcome_id} | no valid pooled estimate |"
                forest_marker = f'data-outcome-id="{fact.outcome_id}"'
                lineage_claim = f'"claim_text":"{fact.outcome_id}"'
                missing = []
                results_disposition = (
                    f"**{label}.** The bound sources reported no valid pooled estimate."
                )
                discussion_disposition = (
                    f"For {label}, the bound sources reported no valid pooled estimate."
                )
                if results_disposition not in section_text["RESULTS"]:
                    missing.append("results")
                if discussion_disposition not in section_text["DISCUSSION"]:
                    missing.append("discussion")
                if not any(line.startswith(table_row) for line in summary_table.splitlines()):
                    missing.append("summary-of-findings")
                if forest_marker not in forest:
                    missing.append("synthesis-inventory")
                if lineage_claim not in claim_lineage_jsonl.decode("utf-8"):
                    missing.append("claim-lineage")
                if missing:
                    issues.append(
                        PublicationVerificationIssue(
                            category="UNREPORTED_SYNTHESIS_DISPOSITION_OMITTED",
                            severity="ERROR",
                            detail=(
                                "The outcome-specific absence disposition is missing from: "
                                f"{', '.join(missing)}."
                            ),
                            fact_ids=(fact.fact_id,),
                        )
                    )
                continue
            effect = _effect_text(fact)
            conclusion_heading = (
                "CONCLUSIONS_AND_RELEVANCE"
                if "CONCLUSIONS_AND_RELEVANCE" in profile.abstract_contract.headings
                else "CONCLUSIONS"
            )
            required_surfaces = {
                "results": section_text["RESULTS"],
                "discussion": section_text["DISCUSSION"],
                "summary-of-findings": summary_table,
                "forest": forest,
            }
            if fact.is_primary or len(synthesis_facts) <= 6:
                required_surfaces["abstract-results"] = _abstract_field(abstract, "RESULTS")
            if fact.analysis_id == primary_analysis_id:
                required_surfaces["abstract-conclusions"] = _abstract_field(
                    abstract, conclusion_heading
                )
            missing = tuple(
                name
                for name, text in required_surfaces.items()
                if not _effect_occurrence_is_exact(text, effect, fact.effect_measure)
            )
            if missing:
                issues.append(
                    PublicationVerificationIssue(
                        category="LOCKED_SYNTHESIS_VALUE_OMITTED",
                        severity="ERROR",
                        detail=(
                            "The sealed synthesis value is absent or changed in: "
                            f"{', '.join(missing)}."
                        ),
                        fact_ids=(fact.fact_id,),
                    )
                )
            if (
                decisions[fact.analysis_id].eligibility == "AUDIT_ONLY"
                and fact.validity != "NOT_REPORTED"
            ):
                abstention = (
                    "contributing Study identity was incomplete, so headline "
                    "promotion remains restricted"
                    if fact.validity == "VALID"
                    else "not presented as a valid pooled benefit or harm"
                )
                if abstention not in manuscript:
                    issues.append(
                        PublicationVerificationIssue(
                            category="INVALID_SYNTHESIS_HEADLINE",
                            severity="ERROR",
                            detail=(
                                "An audit-only synthesis lacks the required abstention statement."
                            ),
                            fact_ids=(fact.fact_id,),
                        )
                    )
        elif fact.kind == "STUDY_OUTCOME":
            if _study_outcome_row(fact) not in supplement:
                issues.append(
                    PublicationVerificationIssue(
                        category="STUDY_OUTCOME_CELL_MISMATCH",
                        severity="ERROR",
                        detail=(
                            "A Study by Outcome row is missing or its disposition/value "
                            "was changed."
                        ),
                        fact_ids=(fact.fact_id,),
                    )
                )
        elif fact.kind == "CERTAINTY":
            matching_rows = [
                line for line in summary_table.splitlines() if f"| {fact.outcome_id} |" in line
            ]
            synthesis = next(
                (
                    item
                    for item in package.facts
                    if item.kind == "SYNTHESIS" and item.outcome_id == fact.outcome_id
                ),
                None,
            )
            contributor_count = (
                ", ".join(synthesis.contributing_study_ids)
                if synthesis and synthesis.contributing_study_ids
                else str(synthesis.contributing_study_count)
                if synthesis and synthesis.contributing_study_count is not None
                else "Not reported"
            )
            contributor_disposition = (
                synthesis.contributor_identity_disposition if synthesis else "NOT_APPLICABLE"
            )
            expected_certainties = (
                f"| {contributor_count} | "
                f"{contributor_disposition} | "
                f"{fact.certainty} | "
                f"{fact.source_review_certainty or 'NOT_REPORTED'} |"
            )
            if len(matching_rows) != 1 or expected_certainties not in matching_rows[0]:
                issues.append(
                    PublicationVerificationIssue(
                        category="LOCKED_CERTAINTY_VALUE_OMITTED",
                        severity="ERROR",
                        detail="Outcome-specific certainty is absent, duplicated, or changed.",
                        fact_ids=(fact.fact_id,),
                    )
                )
    for fact in package.facts:
        if fact.status == "CONFLICT" and fact.kind not in {"SYNTHESIS", "STUDY_OUTCOME"}:
            marker = fact.statement_en.split(".", 1)[0]
            if marker not in surface:
                issues.append(
                    PublicationVerificationIssue(
                        category="SILENT_CONFLICT_OMISSION",
                        severity="ERROR",
                        detail="A conflicting fact is not visible in the delivery.",
                        fact_ids=(fact.fact_id,),
                    )
                )
    known_citations = {reference.citation_key for reference in package.references}
    used_citations = set(re.findall(r"\[([A-Za-z0-9][A-Za-z0-9._-]*)\]", surface))
    invented = used_citations - known_citations
    if invented:
        issues.append(
            PublicationVerificationIssue(
                category="INVENTED_CITATION",
                severity="ERROR",
                detail=f"Delivery cites unknown keys: {', '.join(sorted(invented))}.",
            )
        )
    for index, reference in enumerate(package.references, start=1):
        if f"{index}. [{reference.citation_key}] {reference.display_text}" not in references:
            issues.append(
                PublicationVerificationIssue(
                    category="REFERENCE_INVENTORY_MISMATCH",
                    severity="ERROR",
                    detail=f"Reference entry is absent or changed: {reference.citation_key}.",
                )
            )
    if profile.profile_id == _CONTINUOUS_GENERIC_PROFILE_ID:
        risk_facts = [fact for fact in package.facts if fact.kind == "RISK_OF_BIAS"]
        aggregate_risk = risk_facts and all(
            fact.assessment_level == "REVIEW_AGGREGATE" for fact in risk_facts
        )
        study_level_risk = risk_facts and all(
            fact.assessment_level == "STUDY" for fact in risk_facts
        )
        risk_heading = (
            "### Table 2. Review-level aggregate risk of bias"
            if aggregate_risk
            else (
                "### Table 2. Study-level risk of bias"
                if study_level_risk
                else "### Table 2. Result-specific risk of bias"
            )
        )
        risk_caption = (
            "*Caption: Review-level aggregate risk-of-bias judgment and rationale"
            if aggregate_risk
            else (
                "*Caption: Study-level risk-of-bias judgments and rationales"
                if study_level_risk
                else "*Caption: Result-specific risk-of-bias judgments and rationales"
            )
        )
        supplement_navigation_targets = (
            ("search-strategies-and-dates", "## Search strategies and dates"),
            (
                "complete-study-by-outcome-inventory",
                "## Complete Study by Outcome inventory",
            ),
            ("additional-analysis-dispositions", "## Additional analysis dispositions"),
            ("prisma-checklist-dispositions", "## PRISMA checklist dispositions"),
            (
                "author-responsibility-dispositions",
                "## Author responsibility dispositions",
            ),
        )
        required_continuous_fragments = (
            "## References",
            "## Main Tables and Figures",
            "### Table 1. Study characteristics",
            "*Caption: Source-bound characteristics of every included Study;",
            risk_heading,
            risk_caption,
            "### Table 3. Summary of findings",
            "*Caption: Reported synthesis, validity, headline disposition,",
            "### Figure 1. PRISMA 2020 study flow",
            "*Caption: Study-flow counts from the selection fact.",
            "figures/prisma-flow.md",
            "### Figure 2. Forest plot for ",
            "*Caption: Source-bound forest-plot surface for ",
            "figures/forest-plot.svg",
            "## Supplement Navigation",
            "supplement.md#search-strategies-and-dates",
            "supplement.md#complete-study-by-outcome-inventory",
            "supplement.md#additional-analysis-dispositions",
            "supplement.md#prisma-checklist-dispositions",
            "supplement.md#author-responsibility-dispositions",
            "claim-lineage.jsonl",
        )
        component_lines = tuple(
            line
            for content in (tables, references, prisma_flow)
            for line in content.splitlines()
            if line and not line.startswith("#")
        )
        missing_fragments = tuple(
            fragment for fragment in required_continuous_fragments if fragment not in manuscript
        )
        missing_component_lines = tuple(line for line in component_lines if line not in manuscript)
        supplement_heading_lines = {
            line for line in supplement.splitlines() if line.startswith("#")
        }
        broken_supplement_targets = tuple(
            anchor
            for anchor, heading in supplement_navigation_targets
            if f"supplement.md#{anchor}" not in manuscript
            or heading not in supplement_heading_lines
        )
        if missing_fragments or missing_component_lines or broken_supplement_targets:
            issues.append(
                PublicationVerificationIssue(
                    category="INCOMPLETE_CONTINUOUS_MANUSCRIPT",
                    severity="ERROR",
                    detail=(
                        "Generic manuscript.md does not continuously expose the complete "
                        "references, main tables, figures with captions, or supplement navigation."
                    ),
                )
            )
    _verify_lineage(
        package,
        claim_lineage_jsonl,
        issues,
        surfaces={
            "delivery/manuscript.md": manuscript,
            "delivery/tables.md": tables,
            "delivery/supplement.md": supplement,
            "delivery/references.md": references,
            "delivery/figures/prisma-flow.md": prisma_flow,
            "delivery/figures/forest-plot.svg": forest,
        },
    )
    if compliance.status == "PROFILE_NONCOMPLIANT":
        issues.append(
            PublicationVerificationIssue(
                category="PROFILE_NONCOMPLIANT",
                severity="ERROR",
                detail="One or more deterministic Journal Profile checks failed.",
            )
        )
    conflict_count = sum(fact.status == "CONFLICT" for fact in package.facts)
    return PublicationVerificationReport(
        schema_version="manuscript-publication-verifier.v1",
        package_id=package.package_id,
        profile_id=profile.profile_id,
        verdict="ISSUES_FOUND" if issues else "PASSED",
        issues=tuple(issues),
        checked_fact_count=len(package.facts),
        coverage_count=coverage.covered_count,
        conflict_count=conflict_count,
    )


def _verify_lineage(
    package: ManuscriptFactPackageV2,
    content: bytes,
    issues: list[PublicationVerificationIssue],
    *,
    surfaces: dict[str, str],
) -> None:
    try:
        records = [json.loads(line) for line in content.splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError):
        records = []
    by_fact = {
        record["fact_ids"][0]: record
        for record in records
        if isinstance(record.get("fact_ids"), list) and len(record["fact_ids"]) == 1
    }
    for fact in package.facts:
        record = by_fact.get(fact.fact_id)
        expected_anchors = [anchor.model_dump(mode="json") for anchor in fact.source_anchors]
        expected_terminal_bindings = [
            binding.model_dump(mode="json") for binding in fact.terminal_source_bindings
        ]
        if (
            record is None
            or record.get("source_anchors") != expected_anchors
            or record.get("terminal_source_bindings") != expected_terminal_bindings
            or record.get("upstream_artifact_refs") != list(fact.upstream_artifact_refs)
        ):
            issues.append(
                PublicationVerificationIssue(
                    category="CLAIM_LINEAGE_BINDING_MISMATCH",
                    severity="ERROR",
                    detail="Claim lineage is absent or differs from the sealed fact binding.",
                    fact_ids=(fact.fact_id,),
                )
            )
            continue
        claim_text = record.get("claim_text")
        artifact_paths = record.get("artifact_paths")
        if (
            not isinstance(claim_text, str)
            or not isinstance(artifact_paths, list)
            or not artifact_paths
            or any(
                path not in surfaces or claim_text not in surfaces[path] for path in artifact_paths
            )
        ):
            issues.append(
                PublicationVerificationIssue(
                    category="CLAIM_LINEAGE_SURFACE_MISMATCH",
                    severity="ERROR",
                    detail="Claim lineage does not identify an actual rendered claim surface.",
                    fact_ids=(fact.fact_id,),
                )
            )
        if fact.kind in {"STUDY_OUTCOME", "SYNTHESIS"} and fact.status == "REPORTED":
            if not fact.source_anchors:
                issues.append(
                    PublicationVerificationIssue(
                        category="NUMERIC_FACT_WITHOUT_SOURCE_SPAN",
                        severity="ERROR",
                        detail="A reported numerical fact lacks a Source Span.",
                        fact_ids=(fact.fact_id,),
                    )
                )


def _outcome_label(package: ManuscriptFactPackageV2, outcome_id: str) -> str:
    facts = {fact.outcome_id: fact for fact in package.facts if fact.kind == "OUTCOME"}
    fact = facts.get(outcome_id)
    if fact is None:
        return outcome_id
    duplicate = sum(candidate.label == fact.label for candidate in facts.values()) > 1
    return f"{fact.label} ({fact.timepoint.rstrip('.')})" if duplicate else fact.label


def _section(manuscript: str, heading: str, sections) -> str:
    start = f"## {heading}"
    tail = manuscript.split(start, 1)[1]
    following = [
        tail.find(f"## {section.heading}")
        for section in sections
        if section.heading != heading and tail.find(f"## {section.heading}") >= 0
    ]
    if "## References" in tail:
        following.append(tail.find("## References"))
    return tail[: min(following)] if following else tail


def _abstract_field(abstract: str, heading: str) -> str:
    marker = f"**{heading.replace('_', ' ').title()}:**"
    if marker not in abstract:
        return ""
    tail = abstract.split(marker, 1)[1]
    next_heading = re.search(r"\n\n\*\*[A-Za-z ]+:\*\*", tail)
    return tail[: next_heading.start()] if next_heading else tail


def _effect_occurrence_is_exact(surface: str, effect: str, effect_measure: str) -> bool:
    # Each outcome's locked effect appears exactly once in a target surface; valid effects
    # from other outcomes must not be rejected as extra numbers. Canonical byte comparison
    # of the full artifact still prevents inserting or rewriting unbound values.
    del effect_measure
    return surface.count(effect) == 1


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _result_values(values) -> str:
    if values is None:
        return "Not available"
    if values.result_type == "BINARY":
        return (
            f"{values.intervention_events}/{values.intervention_group_size} versus "
            f"{values.comparator_events}/{values.comparator_group_size}"
        )
    return (
        f"mean {values.intervention_mean:g} (SD {values.intervention_sd:g}), "
        f"n={values.intervention_group_size} versus mean {values.comparator_mean:g} "
        f"(SD {values.comparator_sd:g}), n={values.comparator_group_size}"
    )


def _study_outcome_row(fact) -> str:
    return (
        f"| {_cell(fact.study_id)} | {_cell(fact.outcome_id)} | {fact.result_disposition} | "
        f"{_cell(_result_values(fact.values))} | {_cell(fact.reason or '')} |"
    )


def _effect_text(fact) -> str:
    if fact.estimate is None:
        return "no valid pooled estimate"
    heterogeneity = f"; {fact.heterogeneity}" if fact.heterogeneity else ""
    return (
        f"{fact.effect_measure} {fact.estimate:g} "
        f"(95% CI {fact.ci_lower:g} to {fact.ci_upper:g}{heterogeneity})"
    )

