from __future__ import annotations

import html
import re
from dataclasses import dataclass
from .publication_models import ComplianceCheck, CoverageItem, CoverageManifest, JournalPresentationProfile, ManuscriptFactPackageV2, ProfileComplianceReport
@dataclass(frozen=True, slots=True)
class HeadlineDecision:
    analysis_id: str
    eligibility: str
    reason: str


def decide_headlines(package: ManuscriptFactPackageV2) -> tuple[HeadlineDecision, ...]:
    decisions = []
    for fact in package.facts:
        if fact.kind != "SYNTHESIS":
            continue
        if (
            fact.validity == "VALID"
            and fact.status == "REPORTED"
            and fact.contributor_identity_disposition == "FULL"
        ):
            eligibility = "HEADLINE_ELIGIBLE"
            reason = (
                "The reported synthesis has a closed estimand, source identity, "
                "and validity disposition."
            )
        else:
            eligibility = "AUDIT_ONLY"
            if fact.validity != "VALID":
                reason = (
                    f"The synthesis validity is {fact.validity}; without a human "
                    "adjudication it cannot be presented as a pooled benefit or harm "
                    "headline."
                )
            elif fact.status != "REPORTED":
                reason = (
                    f"The synthesis status is {fact.status}; it cannot be presented as a "
                    "pooled benefit or harm headline."
                )
            else:
                reason = (
                    "The contributing Study identities are not completely reported; the "
                    "estimate remains visible but is not promoted as a headline."
                )
        decisions.append(HeadlineDecision(fact.analysis_id, eligibility, reason))
    return tuple(decisions)


def compile_coverage(
    package: ManuscriptFactPackageV2,
    profile: JournalPresentationProfile,
    *,
    manuscript: str,
    tables: str,
    supplement: str,
    references: str,
    prisma_flow: str,
    forest_svg: bytes,
    section_ids: tuple[str, ...],
    front_matter_modules: tuple[str, ...],
    main_object_ids: tuple[str, ...],
    supplement_object_ids: tuple[str, ...],
    supplement_modules: tuple[str, ...],
    fact_presence: set[str] | None = None,
    semantic_unverified: set[str] | None = None,
) -> CoverageManifest:
    def fact_is_rendered(fact, **surfaces):
        if fact_presence is not None:
            return fact.fact_id in fact_presence
        return _fact_is_rendered(fact, **surfaces)

    items: list[CoverageItem] = []
    for fact in package.facts:
        placement, path, section = _fact_destination(fact.kind, fact.allowed_placements)
        if fact_presence is not None and fact.kind == "NARRATIVE":
            placement, path, section = "MAIN_TEXT", "delivery/manuscript.md", "INTRODUCTION"
        elif fact_presence is not None and fact.kind == "OUTCOME":
            placement, path, section = "SUPPLEMENT", "delivery/supplement.md", "OUTCOME_DEFINITIONS"
        disposition = (
            "CONFLICT_VISIBLE"
            if fact.status == "CONFLICT"
            else "EXPLICIT_ABSENCE"
            if fact.status in {"NOT_REPORTED", "NOT_APPLICABLE", "AWAITING_HUMAN_INPUT"}
            else "COVERED"
        )
        covered = fact_is_rendered(
            fact,
            manuscript=manuscript,
            tables=tables,
            supplement=supplement,
            prisma_flow=prisma_flow,
            forest_svg=forest_svg.decode("utf-8"),
        )
        for requirement in fact.required_for:
            items.append(
                CoverageItem(
                    requirement_id=requirement,
                    requirement_source=(
                        "PRISMA_2020"
                        if requirement.startswith("prisma.")
                        else "PRISMA_S"
                        if requirement.startswith("prisma_s.")
                        else "FACT_PACKAGE"
                    ),
                    fact_ids=(fact.fact_id,),
                    placement=placement,
                    artifact_path=path,
                    section_id=section,
                    disposition=disposition,
                    status="PASSED" if covered else "FAILED",
                )
            )
    author_by_field = {
        fact.field: fact for fact in package.facts if fact.kind == "AUTHOR_RESPONSIBILITY"
    }
    for statement in profile.required_statements:
        fact = author_by_field.get(statement.author_field) if statement.author_field else None
        items.append(
            CoverageItem(
                requirement_id=f"profile.statement.{statement.statement_id.lower()}",
                requirement_source="JOURNAL_PROFILE",
                fact_ids=((fact.fact_id,) if fact else ()),
                placement=statement.placement,
                artifact_path="delivery/supplement.md",
                section_id="AUTHOR_STATEMENTS",
                disposition=(
                    "COVERED"
                    if fact is not None and fact.status == "REPORTED"
                    else "EXPLICIT_ABSENCE"
                ),
                status=("PASSED" if fact is not None and fact.field in supplement else "FAILED"),
            )
        )
    for section in profile.section_contract:
        heading = f"## {section.heading}"
        present = section.section_id in section_ids and heading in manuscript
        items.append(
            CoverageItem(
                requirement_id=f"profile.section.{section.section_id.lower()}",
                requirement_source="JOURNAL_PROFILE",
                fact_ids=(),
                placement="MAIN_TEXT",
                artifact_path="delivery/manuscript.md",
                section_id=section.section_id,
                disposition="COVERED",
                status="PASSED" if present else "FAILED",
            )
        )
    for module in profile.front_matter_modules:
        present = module in front_matter_modules and _front_module_heading(module) in manuscript
        items.append(
            CoverageItem(
                requirement_id=f"profile.front-matter.{module.lower()}",
                requirement_source="JOURNAL_PROFILE",
                fact_ids=(),
                placement="MAIN_TEXT",
                artifact_path="delivery/manuscript.md",
                section_id=module,
                disposition="COVERED",
                status="PASSED" if present else "FAILED",
            )
        )
    for module in profile.supplement_contract.required_modules:
        present = module in supplement_modules and _supplement_module_heading(module) in supplement
        items.append(
            CoverageItem(
                requirement_id=f"profile.supplement.{module.lower()}",
                requirement_source="JOURNAL_PROFILE",
                fact_ids=(),
                placement="SUPPLEMENT",
                artifact_path="delivery/supplement.md",
                section_id=module,
                disposition="COVERED",
                status="PASSED" if present else "FAILED",
            )
        )
    all_objects = set(main_object_ids) | set(supplement_object_ids)
    for object_id in profile.main_figure_table_budget.required_objects:
        present = object_id in all_objects and _object_is_rendered(
            object_id,
            tables=tables,
            supplement=supplement,
            prisma_flow=prisma_flow,
            forest_svg=forest_svg.decode("utf-8"),
        )
        items.append(
            CoverageItem(
                requirement_id=f"profile.object.{object_id.lower()}",
                requirement_source="JOURNAL_PROFILE",
                fact_ids=(),
                placement=("SUPPLEMENT" if object_id in supplement_object_ids else "TABLE_FIGURE"),
                artifact_path=(
                    "delivery/supplement.md"
                    if object_id in supplement_object_ids
                    else "delivery/tables.md"
                ),
                section_id=object_id,
                disposition="COVERED",
                status="PASSED" if present else "FAILED",
            )
        )
    for guideline in profile.reporting_guidelines:
        if guideline == "PRISMA_2020":
            for item_number in range(1, 28):
                owner_facts = tuple(
                    fact
                    for fact in package.facts
                    if any(
                        requirement.startswith(f"prisma.{item_number}.")
                        for requirement in fact.required_for
                    )
                )
                if item_number == 2:
                    passed = "## Abstract" in manuscript and all(
                        _abstract_field_is_present(manuscript, heading)
                        for heading in profile.abstract_contract.headings
                    )
                    owners = ()
                    placement, path, section = (
                        "ABSTRACT",
                        "delivery/manuscript.md",
                        "ABSTRACT",
                    )
                    disposition = "COVERED"
                else:
                    owners = tuple(fact.fact_id for fact in owner_facts)
                    passed = bool(owner_facts) and all(
                        fact_is_rendered(
                            fact,
                            manuscript=manuscript,
                            tables=tables,
                            supplement=supplement,
                            prisma_flow=prisma_flow,
                            forest_svg=forest_svg.decode("utf-8"),
                        )
                        for fact in owner_facts
                    )
                    placement, path, section = (
                        _fact_destination(owner_facts[0].kind, owner_facts[0].allowed_placements)
                        if owner_facts
                        else ("EXPLICIT_DISPOSITION", "delivery/supplement.md", None)
                    )
                    disposition = (
                        "CONFLICT_VISIBLE"
                        if any(fact.status == "CONFLICT" for fact in owner_facts)
                        else "COVERED"
                        if any(fact.status == "REPORTED" for fact in owner_facts)
                        else "EXPLICIT_ABSENCE"
                    )
                items.append(
                    CoverageItem(
                        requirement_id=f"profile.guideline.prisma-2020.item-{item_number}",
                        requirement_source="JOURNAL_PROFILE",
                        fact_ids=owners,
                        placement=placement,
                        artifact_path=path,
                        section_id=section,
                        disposition=disposition,
                        status="PASSED" if passed else "FAILED",
                    )
                )
        elif guideline == "PRISMA_S":
            for item_number in range(1, 17):
                owner_facts = tuple(
                    fact
                    for fact in package.facts
                    if any(
                        requirement.startswith(f"prisma_s.{item_number}.")
                        for requirement in fact.required_for
                    )
                )
                owners = tuple(fact.fact_id for fact in owner_facts)
                passed = bool(owner_facts) and all(
                    fact_is_rendered(
                        fact,
                        manuscript=manuscript,
                        tables=tables,
                        supplement=supplement,
                        prisma_flow=prisma_flow,
                        forest_svg=forest_svg.decode("utf-8"),
                    )
                    for fact in owner_facts
                )
                items.append(
                    CoverageItem(
                        requirement_id=f"profile.guideline.prisma-s.item-{item_number}",
                        requirement_source="JOURNAL_PROFILE",
                        fact_ids=owners,
                        placement="EXPLICIT_DISPOSITION",
                        artifact_path="delivery/supplement.md",
                        section_id=None,
                        disposition="EXPLICIT_ABSENCE",
                        status="PASSED" if passed else "FAILED",
                    )
                )
        elif guideline == "PRISMA_FOR_ABSTRACTS":
            items.append(
                CoverageItem(
                    requirement_id="profile.guideline.prisma-for-abstracts",
                    requirement_source="JOURNAL_PROFILE",
                    fact_ids=(),
                    placement="ABSTRACT",
                    artifact_path="delivery/manuscript.md",
                    section_id="ABSTRACT",
                    disposition="COVERED",
                    status=(
                        "PASSED"
                        if "## Abstract" in manuscript and profile.abstract_contract.structured
                        else "FAILED"
                    ),
                )
            )
        else:
            items.append(
                CoverageItem(
                    requirement_id=f"profile.guideline.{guideline.lower()}",
                    requirement_source="JOURNAL_PROFILE",
                    fact_ids=(),
                    placement="EXPLICIT_DISPOSITION",
                    artifact_path="inputs/journal-profile.json",
                    section_id=None,
                    disposition="EXPLICIT_ABSENCE",
                    status="FAILED",
                )
            )
    items.append(
        CoverageItem(
            requirement_id="profile.citation-style",
            requirement_source="JOURNAL_PROFILE",
            fact_ids=(),
            placement="SUPPLEMENT",
            artifact_path="delivery/references.md",
            section_id=None,
            disposition="COVERED",
            status=(
                "PASSED"
                if _detect_reference_style(package, references) == profile.citation_style
                else "FAILED"
            ),
        )
    )
    if semantic_unverified:
        items = [
            item.model_copy(update={"status": "UNVERIFIED"})
            if item.status == "PASSED" and set(item.fact_ids) & semantic_unverified
            else item
            for item in items
        ]
    covered_count = sum(item.status == "PASSED" for item in items)
    return CoverageManifest(
        schema_version="manuscript-coverage.v1",
        package_id=package.package_id,
        profile_id=profile.profile_id,
        items=tuple(items),
        required_count=len(items),
        covered_count=covered_count,
    )


def evaluate_profile_compliance(
    *,
    package: ManuscriptFactPackageV2,
    profile: JournalPresentationProfile,
    profile_sha256: str,
    title: str,
    abstract: str,
    manuscript: str,
    tables: str,
    supplement: str,
    references: str,
    coverage: CoverageManifest,
    section_ids: tuple[str, ...],
    front_matter_modules: tuple[str, ...],
    main_object_ids: tuple[str, ...],
    supplement_object_ids: tuple[str, ...],
    supplement_modules: tuple[str, ...],
) -> ProfileComplianceReport:
    checks: list[ComplianceCheck] = []
    abstract_words = _word_count(abstract)
    abstract_minimum = profile.abstract_contract.minimum_words
    abstract_maximum = profile.abstract_contract.maximum_words
    abstract_in_range = (abstract_minimum is None or abstract_words >= abstract_minimum) and (
        abstract_maximum is None or abstract_words <= abstract_maximum
    )
    abstract_bounds = (
        f"source-backed range is {abstract_minimum}-{abstract_maximum}"
        if abstract_minimum is not None and abstract_maximum is not None
        else f"source-backed minimum is {abstract_minimum}"
        if abstract_minimum is not None
        else f"source-backed maximum is {abstract_maximum}"
        if abstract_maximum is not None
        else "no source-backed hard range is configured"
    )
    checks.append(
        ComplianceCheck(
            check_id="abstract-word-budget",
            status="PASSED" if abstract_in_range else "FAILED",
            detail=f"Abstract contains {abstract_words} words; {abstract_bounds}.",
        )
    )
    main_words = _word_count(manuscript) - abstract_words
    maximum = profile.word_budget.maximum
    checks.append(
        ComplianceCheck(
            check_id="main-text-word-budget",
            status="PASSED" if maximum is None or main_words <= maximum else "FAILED",
            detail=(
                f"Main manuscript surface contains approximately {main_words} words; "
                f"maximum is {maximum or 'not fixed'}."
            ),
        )
    )
    actual_main_count = len(main_object_ids)
    figure_maximum = profile.main_figure_table_budget.maximum_total
    checks.append(
        ComplianceCheck(
            check_id="main-table-figure-budget",
            status=(
                "PASSED"
                if figure_maximum is None or actual_main_count <= figure_maximum
                else "FAILED"
            ),
            detail=(
                f"Main presentation uses {actual_main_count} table/figure objects; "
                f"maximum is {figure_maximum or 'not fixed'}."
            ),
        )
    )
    checks.append(
        ComplianceCheck(
            check_id="coverage-complete",
            status="PASSED" if coverage.covered_count == coverage.required_count else "FAILED",
            detail=(
                f"Coverage is {coverage.covered_count}/{coverage.required_count}; explicit "
                "absence and visible conflict count as dispositions, not reported facts."
            ),
        )
    )
    full_surface = "\n".join((manuscript, tables, supplement, references))
    forbidden_surface = re.compile(
        r"fact-[A-Za-z0-9]|pub-[A-Za-z0-9]|\[MISSING\]|\bTBD\b|\{\{|"
        r"[\u3400-\u9fff]|Fact Package|Run Folder|Manuscript Specialist|"
        r"Journal Presentation Profile|\bcompiler\b|\bsealed\b|\bV2 closure\b|"
        r"\bsource package\b|\bprogrammed consistency\b|\bCartesian closure\b",
        re.IGNORECASE,
    )
    checks.append(
        ComplianceCheck(
            check_id="forbidden-surface",
            status="FAILED" if forbidden_surface.search(full_surface) else "PASSED",
            detail=(
                "Candidate surface was checked for internal fact tokens, placeholders, "
                "template markers, and Chinese residual text."
            ),
        )
    )
    checks.append(
        ComplianceCheck(
            check_id="title-contract",
            status="PASSED" if title.endswith(profile.title_contract.suffix) else "FAILED",
            detail=f"Title must end with the Profile suffix: {profile.title_contract.suffix}.",
        )
    )
    expected_abstract_labels = tuple(
        f"**{heading.replace('_', ' ').title()}:**"
        for heading in profile.abstract_contract.headings
    )
    abstract_positions = tuple(abstract.find(label) for label in expected_abstract_labels)
    checks.append(
        ComplianceCheck(
            check_id="abstract-heading-contract",
            status=(
                "PASSED"
                if all(position >= 0 for position in abstract_positions)
                and abstract_positions == tuple(sorted(abstract_positions))
                else "FAILED"
            ),
            detail="Structured abstract headings must be present once in Profile order.",
        )
    )
    expected_sections = tuple(section.section_id for section in profile.section_contract)
    checks.append(
        ComplianceCheck(
            check_id="section-contract",
            status="PASSED" if section_ids == expected_sections else "FAILED",
            detail="Rendered IMRaD section inventory must equal the Profile contract.",
        )
    )
    checks.append(
        ComplianceCheck(
            check_id="front-matter-contract",
            status=(
                "PASSED"
                if front_matter_modules == profile.front_matter_modules
                and all(
                    _front_module_heading(module) in manuscript
                    for module in profile.front_matter_modules
                )
                else "FAILED"
            ),
            detail="Rendered front matter modules must equal the Profile contract.",
        )
    )
    expected_supplement = set(profile.supplement_contract.required_modules)
    checks.append(
        ComplianceCheck(
            check_id="supplement-contract",
            status=(
                "PASSED"
                if expected_supplement <= set(supplement_modules)
                and all(
                    _supplement_module_heading(module) in supplement
                    for module in expected_supplement
                )
                else "FAILED"
            ),
            detail="Every required supplement module must be present in the rendered inventory.",
        )
    )
    expected_objects = set(profile.main_figure_table_budget.required_objects)
    actual_objects = set(main_object_ids) | set(supplement_object_ids)
    checks.append(
        ComplianceCheck(
            check_id="required-object-contract",
            status="PASSED" if expected_objects <= actual_objects else "FAILED",
            detail="Every required table or figure must be present in its rendered inventory.",
        )
    )
    checks.append(
        ComplianceCheck(
            check_id="citation-style-contract",
            status=(
                "PASSED"
                if _detect_reference_style(package, references) == profile.citation_style
                else "FAILED"
            ),
            detail=(
                "Detected reference surface is "
                f"{_detect_reference_style(package, references)}; Profile requires "
                f"{profile.citation_style}."
            ),
        )
    )
    reference_count = len(package.references)
    reference_minimum_status = (
        "PASSED"
        if profile.reference_minimum is None or reference_count >= profile.reference_minimum
        else "AWAITING_HUMAN_INPUT"
    )
    checks.append(
        ComplianceCheck(
            check_id="reference-budget",
            status=(
                "FAILED"
                if profile.reference_maximum is not None
                and reference_count > profile.reference_maximum
                else reference_minimum_status
            ),
            detail=(
                f"Reference inventory contains {reference_count} entries; Profile range is "
                f"{profile.reference_minimum or 'not fixed'} to "
                f"{profile.reference_maximum or 'not fixed'}."
            ),
        )
    )
    for index, claim in enumerate(profile.forbidden_claims, start=1):
        checks.append(
            ComplianceCheck(
                check_id=f"profile-forbidden-claim-{index}",
                status="FAILED" if claim.lower() in full_surface.lower() else "PASSED",
                detail=f"Profile-forbidden claim was checked: {claim}.",
            )
        )
    author_by_field = {
        fact.field: fact for fact in package.facts if fact.kind == "AUTHOR_RESPONSIBILITY"
    }
    for statement in profile.required_statements:
        fact = author_by_field.get(statement.author_field) if statement.author_field else None
        status = (
            "PASSED" if fact is not None and fact.status == "REPORTED" else "AWAITING_HUMAN_INPUT"
        )
        checks.append(
            ComplianceCheck(
                check_id=f"author-statement-{statement.statement_id.lower()}",
                status=status,
                detail=(
                    f"{statement.statement_id} is source-bound to named researcher input."
                    if status == "PASSED"
                    else (
                        f"{statement.statement_id} awaits named researcher input and was "
                        "not invented."
                    )
                ),
            )
        )
    if any(check.status == "FAILED" for check in checks):
        status = "PROFILE_NONCOMPLIANT"
    elif any(check.status == "AWAITING_HUMAN_INPUT" for check in checks):
        status = "NOT_READY_FOR_HUMAN_FINALIZATION"
    else:
        status = "PROFILE_COMPLETE"
    return ProfileComplianceReport(
        schema_version="manuscript-profile-compliance.v1",
        package_id=package.package_id,
        profile_id=profile.profile_id,
        profile_sha256=profile_sha256,
        checks=tuple(checks),
        status=status,
    )


def _fact_destination(kind: str, allowed: tuple[str, ...]) -> tuple[str, str, str | None]:
    if kind == "SELECTION_FLOW":
        return "TABLE_FIGURE", "delivery/figures/prisma-flow.md", "RESULTS"
    if kind in {"STUDY", "RISK_OF_BIAS", "CERTAINTY"}:
        return "TABLE_FIGURE", "delivery/tables.md", "RESULTS"
    if kind == "STUDY_OUTCOME":
        return "SUPPLEMENT", "delivery/supplement.md", "STUDY_OUTCOME_INVENTORY"
    if kind in {
        "SEARCH_SOURCE",
        "ANALYSIS_AVAILABILITY",
        "AUTHOR_RESPONSIBILITY",
        "PRISMA_CHECKLIST_DISPOSITION",
    }:
        return "SUPPLEMENT", "delivery/supplement.md", None
    if kind == "SYNTHESIS":
        return "MAIN_TEXT", "delivery/manuscript.md", "RESULTS"
    if kind == "OUTCOME":
        return "MAIN_TEXT", "delivery/manuscript.md", "RESULTS"
    if kind == "REVIEW_METHOD":
        return "MAIN_TEXT", "delivery/manuscript.md", "METHODS"
    if kind == "INTERPRETATION":
        return "MAIN_TEXT", "delivery/manuscript.md", "DISCUSSION"
    placement = "MAIN_TEXT" if "MAIN_TEXT" in allowed else allowed[0]
    return placement, "delivery/manuscript.md", None


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'\-]*\b", text))


def _abstract_field_is_present(manuscript: str, heading: str) -> bool:
    marker = f"**{heading.replace('_', ' ').title()}:**"
    return manuscript.count(marker) == 1


def _front_module_heading(module: str) -> str:
    return {
        "KEY_POINTS": "## Key Points",
        "WHAT_IS_ALREADY_KNOWN": "## What is already known on this topic",
        "WHAT_THIS_STUDY_ADDS": "## What this study adds",
    }[module]


def _supplement_module_heading(module: str) -> str:
    return {
        "PRISMA_FLOW": "## PRISMA flow",
        "SEARCH_STRATEGIES": "## Search strategies and dates",
        "STUDY_OUTCOME_INVENTORY": "## Complete Study by Outcome inventory",
        "ADDITIONAL_ANALYSIS_DISPOSITIONS": "## Additional analysis dispositions",
        "AUTHOR_INPUT_DISPOSITIONS": "## Author responsibility dispositions",
    }[module]


def _object_is_rendered(
    object_id: str,
    *,
    tables: str,
    supplement: str,
    prisma_flow: str,
    forest_svg: str,
) -> bool:
    return {
        "PRISMA_FLOW": "# PRISMA 2020 flow" in prisma_flow,
        "STUDY_CHARACTERISTICS": "## Table 1. Study characteristics" in tables,
        "RISK_OF_BIAS": any(
            heading in tables
            for heading in (
                "## Table 2. Review-level aggregate risk of bias",
                "## Table 2. Result-specific risk of bias",
                "## Table 2. Study-level risk of bias",
            )
        ),
        "FOREST_PLOT": forest_svg.startswith("<svg"),
        "SUMMARY_OF_FINDINGS": "## Table 3. Summary of findings" in tables,
    }[object_id]


def _fact_is_rendered(
    fact,
    *,
    manuscript: str,
    tables: str,
    supplement: str,
    prisma_flow: str,
    forest_svg: str,
) -> bool:
    if fact.kind == "NARRATIVE":
        marker = "systematic review" if fact.topic == "REVIEW_IDENTITY" else fact.statement_en
        return marker.lower() in manuscript.lower()
    if fact.kind == "SEARCH_SOURCE":
        expected = (
            f"- **{fact.source_name}:** last search date: "
            f"{fact.last_search_date or 'not reported'}; strategy: "
            f"{fact.strategy or 'not reported in the available sources'}; "
            f"disposition: {fact.strategy_disposition}."
        )
        return supplement.splitlines().count(expected) == 1
    if fact.kind == "SELECTION_FLOW":
        lines = set(prisma_flow.splitlines())
        unknown = "Not reported"
        return all(
            line in lines
            for line in (
                "Records identified: "
                f"{fact.records_identified if fact.records_identified is not None else unknown}",
                "Duplicates removed: "
                f"{fact.duplicates_removed if fact.duplicates_removed is not None else unknown}",
                "Records screened: "
                f"{fact.records_screened if fact.records_screened is not None else unknown}",
                "Reports sought: "
                f"{fact.reports_sought if fact.reports_sought is not None else unknown}",
                "Reports not retrieved: "
                f"{fact.reports_not_retrieved if fact.reports_not_retrieved is not None else unknown}",  # noqa: E501
                "Reports assessed: "
                f"{fact.reports_assessed if fact.reports_assessed is not None else unknown}",
                "Reports excluded: "
                f"{fact.reports_excluded if fact.reports_excluded is not None else unknown}",
                f"Studies included: {fact.studies_included}",
                f"Reports included: {fact.reports_included}",
            )
        )
    if fact.kind == "STUDY":
        row = (
            f"| {_cell(fact.study_id)} | {_cell(fact.design or 'Not reported')} | "
            f"{fact.sample_size or 'Not reported'} | "
            f"{_cell(fact.population or 'Not reported')} | "
            f"{_cell(fact.intervention or 'Not reported')} | "
            f"{_cell(fact.comparator or 'Not reported')} | "
            f"{_cell(fact.follow_up or 'Not reported')} | "
            f"{_cell(fact.source_characteristics or 'Not reported')} |"
        )
        return tables.splitlines().count(row) == 1
    if fact.kind == "STUDY_OUTCOME":
        row = (
            f"| {_cell(fact.study_id)} | {_cell(fact.outcome_id)} | "
            f"{fact.result_disposition} | {_cell(_result_values(fact.values))} | "
            f"{_cell(fact.reason or '')} |"
        )
        return supplement.splitlines().count(row) == 1
    if fact.kind == "SYNTHESIS":
        effect = _effect_text(fact)
        table_row = f"| {_cell(fact.outcome_id)} | {_cell(effect)} |"
        forest_marker = f'data-outcome-id="{html.escape(fact.outcome_id)}"'
        return (
            effect in manuscript
            and sum(line.startswith(table_row) for line in tables.splitlines()) == 1
            and forest_marker in forest_svg
        )
    if fact.kind == "OUTCOME":
        return fact.statement_en in manuscript or fact.statement_en in supplement
    if fact.kind == "REVIEW_METHOD":
        return fact.description in manuscript
    if fact.kind == "INTERPRETATION":
        return fact.statement_en in manuscript
    if fact.kind == "RISK_OF_BIAS":
        row = (
            f"| {_cell(fact.study_id or 'All included studies')} | "
            f"{_cell(fact.outcome_id or 'Review-level aggregate')} | "
            f"{_cell(fact.judgment)} | {_cell(fact.rationale)} |"
        )
        return tables.splitlines().count(row) == 1
    if fact.kind == "CERTAINTY":
        prefix = f"| {_cell(fact.outcome_id)} | "
        suffix = (
            f"| {_cell(fact.certainty)} | {_cell(fact.source_review_certainty or 'NOT_REPORTED')} |"
        )
        return (
            sum(line.startswith(prefix) and line.endswith(suffix) for line in tables.splitlines())
            == 1
        )
    if fact.kind in {
        "ANALYSIS_AVAILABILITY",
        "AUTHOR_RESPONSIBILITY",
        "PRISMA_CHECKLIST_DISPOSITION",
    }:
        return fact.statement_en in supplement
    return False


def _detect_reference_style(package: ManuscriptFactPackageV2, references: str) -> str:
    expected = ["# References"]
    expected.extend(
        f"{index}. [{reference.citation_key}] {reference.display_text}"
        for index, reference in enumerate(package.references, start=1)
    )
    observed = [line for line in references.splitlines() if line]
    return "SOURCE_BOUND_KEYED" if observed == expected else "UNRECOGNIZED"


def _effect_text(fact) -> str:
    if fact.estimate is None:
        return "no valid pooled estimate"
    heterogeneity = f"; {fact.heterogeneity}" if fact.heterogeneity else ""
    return (
        f"{fact.effect_measure} {fact.estimate:g} "
        f"(95% CI {fact.ci_lower:g} to {fact.ci_upper:g}{heterogeneity})"
    )


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


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


