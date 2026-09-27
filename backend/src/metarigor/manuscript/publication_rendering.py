from __future__ import annotations

import html
from dataclasses import dataclass
from .models import canonical_json
from .publication_coverage import HeadlineDecision, decide_headlines
from .publication_models import JournalPresentationProfile, ManuscriptFactPackageV2, SynthesisPublicationFact
_CONTINUOUS_GENERIC_PROFILE_ID = "generic_prisma_2020_meta_analysis_v2"


@dataclass(frozen=True, slots=True)
class RenderedPublication:
    title: str
    abstract: str
    manuscript: str
    tables: str
    supplement: str
    references: str
    prisma_flow: str
    forest_svg: bytes
    claim_lineage_jsonl: bytes
    main_table_figure_count: int
    section_ids: tuple[str, ...]
    front_matter_modules: tuple[str, ...]
    main_object_ids: tuple[str, ...]
    supplement_object_ids: tuple[str, ...]
    supplement_modules: tuple[str, ...]


def render_publication(
    package: ManuscriptFactPackageV2, profile: JournalPresentationProfile
) -> RenderedPublication:
    decisions = decide_headlines(package)
    title = _title(package, profile)
    abstract = _abstract(package, profile, decisions)
    front = _front_matter(package, profile, decisions)
    tables = _tables(package, decisions)
    supplement = _supplement(package, profile, decisions)
    references = _references(package)
    prisma_flow = _prisma_flow(package)
    forest_svg = _forest_svg(package, decisions)
    manuscript = _manuscript(
        package,
        profile,
        title,
        abstract,
        front,
        decisions,
        tables=tables,
        references=references,
        prisma_flow=prisma_flow,
    )
    lineage = _lineage(package, profile)
    main_object_ids = (
        "STUDY_CHARACTERISTICS",
        "RISK_OF_BIAS",
        "FOREST_PLOT",
        "SUMMARY_OF_FINDINGS",
    )
    supplement_object_ids: tuple[str, ...] = ()
    if profile.supplement_contract.prisma_flow_placement == "MAIN":
        main_object_ids = (*main_object_ids, "PRISMA_FLOW")
    else:
        supplement_object_ids = ("PRISMA_FLOW",)
    supplement_modules = (
        "SEARCH_STRATEGIES",
        "STUDY_OUTCOME_INVENTORY",
        "ADDITIONAL_ANALYSIS_DISPOSITIONS",
        "AUTHOR_INPUT_DISPOSITIONS",
        *supplement_object_ids,
    )
    return RenderedPublication(
        title=title,
        abstract=abstract,
        manuscript=manuscript,
        tables=tables,
        supplement=supplement,
        references=references,
        prisma_flow=prisma_flow,
        forest_svg=forest_svg,
        claim_lineage_jsonl=lineage,
        main_table_figure_count=len(main_object_ids),
        section_ids=tuple(section.section_id for section in profile.section_contract),
        front_matter_modules=profile.front_matter_modules,
        main_object_ids=main_object_ids,
        supplement_object_ids=supplement_object_ids,
        supplement_modules=supplement_modules,
    )


def _title(package: ManuscriptFactPackageV2, profile: JournalPresentationProfile) -> str:
    core = (
        f"{package.scope.intervention} versus {package.scope.comparator} for "
        f"{package.scope.population}"
    )
    return f"{core}: {profile.title_contract.suffix}"


def _abstract(
    package: ManuscriptFactPackageV2,
    profile: JournalPresentationProfile,
    decisions: tuple[HeadlineDecision, ...],
) -> str:
    objective = _narrative(package, "OBJECTIVE")
    eligibility = _narrative(package, "ELIGIBILITY")
    limitation = _narrative(package, "LIMITATION")
    searches = [fact for fact in package.facts if fact.kind == "SEARCH_SOURCE"]
    selection = next(fact for fact in package.facts if fact.kind == "SELECTION_FLOW")
    synthesis = _primary_synthesis(package)
    result = _synthesis_summary(package, decisions, include_details=False)
    primary_result = _primary_synthesis_summary(package, decisions, include_details=False)
    if len(_syntheses(package)) > 6:
        absent = sum(fact.validity == "NOT_REPORTED" for fact in _syntheses(package))
        result = (
            f"{primary_result} The Results and summary-of-findings table retain "
            f"{len(_syntheses(package)) - 1} additional outcome-specific syntheses"
            + (f", including {absent} without a valid pooled estimate" if absent else "")
            + "."
        )
    sources = ", ".join(fact.source_name for fact in searches)
    complete = package.fact_scope == "COMPLETE_SOURCE_SUPPORTED"
    method_facts = [fact for fact in package.facts if fact.kind == "REVIEW_METHOD"]
    reported_domains = [
        fact.method_domain.lower().replace("_", "-")
        for fact in method_facts
        if fact.status == "REPORTED"
    ]
    unavailable_domain_count = sum(fact.status != "REPORTED" for fact in method_facts)
    method_summary = (
        "Source-bound review methods covered "
        + ", ".join(reported_domains)
        + (
            f"; {unavailable_domain_count} method domains remained explicitly unavailable."
            if unavailable_domain_count
            else "."
        )
        if method_facts
        else (
            "The available sources did not fully report search strategies, selection "
            "procedures, data collection methods, Study characteristics, result-specific "
            "risk-of-bias judgments, or additional analyses; these gaps remain explicitly "
            "unreported."
        )
    )
    certainty_summary = _certainty_summary(package)
    contents = {
        "BACKGROUND": (
            "Systematic reviews can inform decisions only when study identity, outcome definitions, numerical results, and unresolved source conflicts remain traceable. This review therefore separates reported findings from explicit absence and conflict dispositions across every prespecified outcome."
        ),
        "IMPORTANCE": (
            "Clinical interpretation of a meta-analysis depends on complete reporting of study selection, outcome-specific evidence, risk of bias, certainty, and synthesis validity. This source-bound review preserves those elements without allowing presentation requirements to alter the research facts."
        ),
        "OBJECTIVE": objective,
        "METHODS": (
            f"The accepted protocol governed eligibility and analysis intent. Reported information sources included {sources}. Results were sought for {len(package.outcome_ids)} prespecified outcomes in {len(package.study_ids)} included studies."
            f" {method_summary}"
        ),
        "DATA_SOURCES": (
            f"Reported information sources included {sources}. Unavailable source-specific strategies and dates are identified in the supplement."
        ),
        "STUDY_SELECTION": (
            f"{eligibility} The review includes {selection.studies_included} studies represented by {selection.reports_included} reports. The PRISMA flow reports every available count and marks unavailable counts explicitly."
        ),
        "DATA_EXTRACTION_AND_SYNTHESIS": (
            "Reported Study-level numerical results were collected for the prespecified outcomes. Missing results and unresolved source discrepancies are identified in the supplement."
            + (
                " Source-bound data-collection, risk-of-bias, and synthesis procedures are reported at the level supported by the review."
                if complete
                else " Complete data collection methods and result-specific risk-of-bias procedures were not reported in the available sources."
            )
        ),
        "MAIN_OUTCOMES_AND_MEASURES": (
            f"The review contains {len(package.outcome_ids)} prespecified outcomes. The reported synthesis used {synthesis.effect_measure}; its analysis population, timepoint, contributing studies, confidence interval, heterogeneity, and validity are reported separately."
        ),
        "RESULTS": (
            f"The review included {len(package.study_ids)} studies and {len(package.outcome_ids)} outcomes. {result} {certainty_summary}"
        ),
        "LIMITATIONS": limitation,
        "CONCLUSIONS": (
            f"{primary_result} Interpretation accounts for outcome-specific certainty, source conflicts, and the reported limitations."
            if complete
            else f"{primary_result} Interpretation should remain conditional on the reported certainty and all visible source, selection, risk-of-bias, and analysis dispositions. No missing author statement or unresolved synthesis was repaired by the writer."
        ),
        "CONCLUSIONS_AND_RELEVANCE": (
            f"{primary_result} Clinical relevance should be judged with the source-bound certainty and limitations."
            if complete
            else f"{primary_result} Clinical relevance must be judged with the source-bound certainty assessment and visible limitations; this presentation is not a substitute for named researcher finalization."
        ),
        "DESIGN": (
            "Systematic review and pairwise meta-analysis using an accepted protocol. Eligibility, information sources, Study results, synthesis validity, risk of bias, and certainty are reported separately at their source-supported level."
            if complete
            else "Systematic review and pairwise meta-analysis using a prespecified protocol. Eligibility, information sources, Study results, synthesis validity, risk of bias, and certainty are reported separately when available. Complete search strategies, selection procedures, data collection methods, and Study-level characteristics were not reported in the available sources."
        ),
        "ELIGIBILITY_CRITERIA": eligibility,
        "MAIN_OUTCOME_MEASURES": (
            f"Prespecified outcomes were {', '.join(_outcome_labels(package))}. Reported values, correct abstentions, source conflicts, and synthesis validity were retained as distinct facts."
            if complete
            else f"Prespecified outcomes were {', '.join(_outcome_labels(package))}. Reported values, correct abstentions, source conflicts, and synthesis validity were retained as distinct facts. For each outcome, the results identify how many included Studies had an available numerical result, no reported result, an unavailable source, or an unresolved reference discrepancy."
        ),
        "REGISTRATION": (
            _optional_narrative(
                package,
                "REGISTRATION",
                "A registration identifier was not reported in the available sources reviewed for this manuscript and remains unavailable.",
            )
        ),
    }
    return "\n\n".join(
        f"**{heading.replace('_', ' ').title()}:** {contents[heading]}"
        for heading in profile.abstract_contract.headings
    )


def _front_matter(
    package: ManuscriptFactPackageV2,
    profile: JournalPresentationProfile,
    decisions: tuple[HeadlineDecision, ...],
) -> str:
    if not profile.front_matter_modules:
        return ""
    blocks = []
    synthesis_summary = (
        _primary_synthesis_summary(package, decisions)
        + f" The full Results retain {len(_syntheses(package)) - 1} additional outcome-specific syntheses."
        if len(_syntheses(package)) > 8
        else _synthesis_summary(package, decisions)
    )
    for module in profile.front_matter_modules:
        if module == "KEY_POINTS":
            blocks.append(
                "## Key Points\n\n**Question:** What does the reported evidence show for the prespecified comparison and outcomes?\n\n"
                f"**Findings:** {synthesis_summary} The inventory contains {len(package.study_ids)} studies and {len(package.outcome_ids)} outcomes.\n\n"
                "**Meaning:** Interpretation must retain the certainty assessment and all visible absence or conflict dispositions."
            )
        elif module == "WHAT_IS_ALREADY_KNOWN":
            blocks.append(
                "## What is already known on this topic\n\n"
                f"- The comparison concerns {package.scope.intervention} and {package.scope.comparator}.\n"
                "- Published synthesis alone cannot resolve missing primary-source values or conflicting estimands."
            )
        elif module == "WHAT_THIS_STUDY_ADDS":
            blocks.append(
                "## What this study adds\n\n"
                f"- The review exposes a complete {len(package.study_ids)}-Study by {len(package.outcome_ids)}-outcome inventory, including explicit absences.\n"
                f"- {synthesis_summary}"
            )
    return "\n\n".join(blocks)


def _manuscript(
    package,
    profile,
    title,
    abstract,
    front,
    decisions,
    *,
    tables: str,
    references: str,
    prisma_flow: str,
) -> str:
    synthesis = _primary_synthesis(package)
    decision = next(item for item in decisions if item.analysis_id == synthesis.analysis_id)
    selection = next(fact for fact in package.facts if fact.kind == "SELECTION_FLOW")
    unavailable_cells = sum(
        fact.result_disposition != "EXTRACTED"
        for fact in package.facts
        if fact.kind == "STUDY_OUTCOME"
    )
    methods_expansion = _methods_expansion(package)
    results_expansion = _results_expansion(
        package,
        compact=(profile.word_budget.maximum is not None and profile.word_budget.maximum <= 3000),
    )
    discussion_expansion = _discussion_expansion(package, synthesis, decision)
    sections = [f"# {title}", "## Abstract\n\n" + abstract]
    if front:
        sections.append(front)
    sections.extend(
        [
            f"## {_section_heading(profile, 'INTRODUCTION')}\n\n"
            "This systematic review and pairwise meta-analysis evaluates the prespecified "
            "comparison and outcome inventory. "
            + _narrative(package, "CONTEXT")
            + " Complete reporting is necessary because a polished narrative can otherwise conceal unavailable values, incompatible estimands, or unresolved source discrepancies. "
            + _narrative(package, "OBJECTIVE"),
            f"## {_section_heading(profile, 'METHODS')}\n\n" + methods_expansion,
            f"## {_section_heading(profile, 'RESULTS')}\n\n### Study selection and characteristics\n\n"
            f"The review contains {selection.studies_included} included studies represented by {selection.reports_included} reports. Available and unavailable flow counts are shown without inference in the PRISMA flow diagram. Study characteristics are presented in Table 1.\n\n"
            "### Study results and synthesis\n\n"
            f"Across the prespecified outcomes, {unavailable_cells} Study by Outcome results were unavailable or retained a reference conflict. They remain listed in the supplement. All source-bound published syntheses are reported below.\n\n"
            "### Risk of bias and additional analyses\n\n"
            + _risk_and_additional_analysis_summary(package),
            results_expansion,
            f"## {_section_heading(profile, 'DISCUSSION')}\n\n### Principal findings\n\n"
            f"{_synthesis_summary(package, decisions)} The results must be read together with their outcome-specific certainty assessments and the complete Study by Outcome inventory.\n\n"
            "### Strengths and limitations\n\n"
            + _narrative(package, "LIMITATION")
            + _limitation_suffix(package)
            + "\n\n### Implications\n\nThe evidence can inform clinical and research discussion only within the reported outcome definitions, timepoints, analysis populations, certainty, and validity disposition.",
            discussion_expansion,
            _interpretation_expansion(package),
        ]
    )
    if profile.profile_id == _CONTINUOUS_GENERIC_PROFILE_ID:
        sections.extend(
            [
                _embedded_references(references),
                _embedded_main_tables_and_figures(
                    package,
                    decisions,
                    tables=tables,
                    prisma_flow=prisma_flow,
                ),
                _supplement_navigation(),
            ]
        )
    else:
        sections.append("## References\n\nSee the separate reference list.")
    return "\n\n".join(sections).strip() + "\n"


def _embedded_references(references: str) -> str:
    return references.replace("# References", "## References", 1).strip()


def _embedded_main_tables_and_figures(
    package: ManuscriptFactPackageV2,
    decisions: tuple[HeadlineDecision, ...],
    *,
    tables: str,
    prisma_flow: str,
) -> str:
    synthesis = _primary_synthesis(package)
    decision = next(item for item in decisions if item.analysis_id == synthesis.analysis_id)
    risk_title, risk_caption = _risk_table_presentation(package)
    embedded_tables = tables.removeprefix("# Tables\n").strip()
    embedded_tables = embedded_tables.replace(
        "## Table 1. Study characteristics",
        "### Table 1. Study characteristics\n\n"
        "*Caption: Source-bound characteristics of every included Study; unavailable fields "
        "are shown as not reported rather than inferred.*\n",
    ).replace(
        f"## Table 2. {risk_title}",
        f"### Table 2. {risk_title}\n\n"
        f"*Caption: {risk_caption} retained from the evidence record.*\n",
    ).replace(
        "## Table 3. Summary of findings",
        "### Table 3. Summary of findings\n\n"
        "*Caption: Reported synthesis, validity, headline disposition, and outcome-specific "
        "certainty; explicit absence is not converted into a reported result.*\n",
    )
    embedded_flow = prisma_flow.removeprefix("# PRISMA 2020 flow\n\n").strip()
    return (
        "## Main Tables and Figures\n\n"
        f"{embedded_tables}\n\n"
        "### Figure 1. PRISMA 2020 study flow\n\n"
        "*Caption: Study-flow counts from the selection fact. Unavailable counts remain "
        "explicitly not reported and are not displayed as zero.*\n\n"
        f"{embedded_flow}\n\n"
        "[Open the standalone PRISMA flow artifact](figures/prisma-flow.md)\n\n"
        "### Figure 2. Forest plot for reported estimates and outcome absence inventory\n\n"
        "![Forest plot for reported estimates and outcome absence inventory](figures/forest-plot.svg)\n\n"
        f"*Caption: Source-bound forest-plot surface for "
        f"{sum(fact.validity != 'NOT_REPORTED' for fact in _syntheses(package))} reported estimates and "
        f"{sum(fact.validity == 'NOT_REPORTED' for fact in _syntheses(package))} outcomes without a valid pooled estimate."
        f" Headline disposition: {decision.eligibility}. {decision.reason}*"
    )


def _supplement_navigation() -> str:
    return (
        "## Supplement Navigation\n\n"
        "The auditable supplement remains a separate, immutable package component. Its "
        "sections can be opened directly:\n\n"
        "- [Search strategies and dates](supplement.md#search-strategies-and-dates)\n"
        "- [Complete Study by Outcome inventory](supplement.md#complete-study-by-outcome-inventory)\n"
        "- [Additional analysis dispositions](supplement.md#additional-analysis-dispositions)\n"
        "- [PRISMA checklist dispositions](supplement.md#prisma-checklist-dispositions)\n"
        "- [Author responsibility dispositions](supplement.md#author-responsibility-dispositions)\n"
        "- [Claim lineage and Source Span bindings](claim-lineage.jsonl)"
    )


def _methods_expansion(package) -> str:
    searches = [fact for fact in package.facts if fact.kind == "SEARCH_SOURCE"]
    analysis = [fact for fact in package.facts if fact.kind == "ANALYSIS_AVAILABILITY"]
    synthesis = _primary_synthesis(package)
    methods = [fact for fact in package.facts if fact.kind == "REVIEW_METHOD"]
    missing_methods = [
        fact.statement_en
        for fact in package.facts
        if fact.kind == "PRISMA_CHECKLIST_DISPOSITION"
        and fact.guideline == "PRISMA_2020"
        and 7 <= fact.item_number <= 15
    ]
    if methods:
        method_by_domain = {fact.method_domain: fact for fact in methods}
        return "\n\n".join(
            (
                "### Protocol, eligibility, and registration\n\n"
                + _with_citations(
                    method_by_domain["ELIGIBILITY"],
                    _method_text(method_by_domain["ELIGIBILITY"]),
                )
                + " "
                + _narrative(package, "ELIGIBILITY")
                + " "
                + _optional_narrative(package, "REGISTRATION", ""),
                "### Information sources and search\n\n"
                + " ".join(_with_citations(fact, fact.statement_en) for fact in searches),
                "### Study selection and data collection\n\n"
                + _with_citations(
                    method_by_domain["SELECTION_PROCESS"],
                    _method_text(method_by_domain["SELECTION_PROCESS"]),
                )
                + " "
                + _with_citations(
                    method_by_domain["DATA_COLLECTION"],
                    _method_text(method_by_domain["DATA_COLLECTION"]),
                ),
                "### Risk of bias and certainty\n\n"
                + _with_citations(
                    method_by_domain["RISK_OF_BIAS"],
                    _method_text(method_by_domain["RISK_OF_BIAS"]),
                )
                + " "
                + _with_citations(
                    method_by_domain["CERTAINTY"],
                    _method_text(method_by_domain["CERTAINTY"]),
                ),
                "### Effect measures and synthesis\n\n"
                + " ".join(
                    _with_citations(
                        method_by_domain[domain], _method_text(method_by_domain[domain])
                    )
                    for domain in (
                        "EFFECT_MEASURE",
                        "SYNTHESIS_MODEL",
                        "HETEROGENEITY",
                        "SUBGROUP",
                        "SENSITIVITY",
                        "REPORTING_BIAS",
                        "SOFTWARE",
                    )
                ),
            )
        )
    return "\n\n".join(
        (
            "### Protocol and eligibility\n\n" + _narrative(package, "ELIGIBILITY"),
            "### Information sources and search\n\n"
            + " ".join(fact.statement_en for fact in searches),
            "### Review methods not reported in the available sources\n\n"
            + " ".join(missing_methods),
            "### Synthesis and additional analyses\n\n"
            + synthesis.statement_en
            + " "
            + " ".join(fact.statement_en for fact in analysis),
        )
    )


def _method_text(fact) -> str:
    if fact.method_basis == "PRESPECIFIED_PROTOCOL":
        return f"The accepted protocol specified: {fact.description}"
    return fact.description


def _results_expansion(package, *, compact: bool) -> str:
    studies = [fact for fact in package.facts if fact.kind == "STUDY"]
    cells = [fact for fact in package.facts if fact.kind == "STUDY_OUTCOME"]
    lines = ["### Included Studies", ""]
    if compact:
        study_labels = ", ".join(_study_label(fact.study_id) for fact in studies)
        citations = " ".join(f"[{key}]" for fact in studies for key in fact.citation_keys)
        characteristic_sentence = (
            "Their source-bound characteristics are reported in Table 1."
            if all(fact.status == "REPORTED" for fact in studies)
            else "Complete source-bound Study characteristics were unavailable and remain explicit in Table 1."
        )
        lines.extend(
            [
                f"The included Studies were {study_labels}. {characteristic_sentence} {citations}".strip(),
                "",
            ]
        )
    else:
        for fact in studies:
            citation = " ".join(f"[{key}]" for key in fact.citation_keys)
            statement = fact.statement_en.replace(
                fact.study_id, _study_label(fact.study_id), 1
            )
            lines.append(
                f"**{_study_label(fact.study_id)}.** {statement} {citation}".strip()
            )
            lines.append("")
    outcome_facts = {fact.outcome_id: fact for fact in package.facts if fact.kind == "OUTCOME"}
    large_compact_inventory = compact and len(package.outcome_ids) > 10
    if outcome_facts and large_compact_inventory:
        lines.extend(
            [
                "### Outcome definitions",
                "",
                f"Definitions and timepoints for all {len(package.outcome_ids)} outcomes are retained in the supplement; the Summary of Findings table links each outcome to its synthesis and certainty disposition.",
                "",
            ]
        )
    elif outcome_facts:
        lines.extend(["### Outcome definitions", ""])
        for outcome_id in package.outcome_ids:
            fact = outcome_facts[outcome_id]
            lines.extend(
                [
                    f"**{fact.label}.** {_with_citations(fact, fact.statement_en)}",
                    "",
                ]
            )
    lines.extend(["### Outcome reporting", ""])
    if large_compact_inventory:
        extracted_count = sum(
            fact.result_disposition == "EXTRACTED" for fact in cells
        )
        lines.extend(
            [
                f"The supplement retains all {len(cells)} Study by Outcome cells: {extracted_count} contained complete extracted values and {len(cells) - extracted_count} retained an explicit absence, unavailable source, or conflict disposition.",
                "",
            ]
        )
    for outcome in (() if large_compact_inventory else package.outcome_ids):
        outcome_cells = [fact for fact in cells if fact.outcome_id == outcome]
        if not outcome_cells:
            lines.extend(
                [
                    f"**{_outcome_label(package, outcome)}.** This outcome was reported only as a review-level synthesis; no Study-level extraction target was defined.",
                    "",
                ]
            )
            continue
        extracted = [fact for fact in outcome_cells if fact.result_disposition == "EXTRACTED"]
        not_reported = [fact for fact in outcome_cells if fact.result_disposition == "NOT_REPORTED"]
        unavailable = [
            fact for fact in outcome_cells if fact.result_disposition == "SOURCE_UNAVAILABLE"
        ]
        conflicts = [
            fact for fact in outcome_cells if fact.result_disposition == "REFERENCE_CONFLICT"
        ]
        if compact:
            value_sentence = (
                "Exact group values are reported in the supplement."
                if extracted
                else "No Study supplied a complete extractable result under the prespecified value contract."
            )
        else:
            values = "; ".join(
                f"{_study_label(fact.study_id)}: {_result_values(fact.values)}"
                for fact in extracted[:4]
            )
            remainder = len(extracted) - min(len(extracted), 4)
            value_sentence = (
                f"Extracted Study results included {values}"
                + (
                    f", with {remainder} additional extracted "
                    f"{'result' if remainder == 1 else 'results'} in the supplement"
                    if remainder
                    else ""
                )
                + "."
                if extracted
                else "No Study supplied a complete extractable result under the prespecified value contract."
            )
        label = _outcome_label(package, outcome)
        disposition_parts = []
        if not_reported:
            disposition_parts.append(
                "1 result was not reported"
                if len(not_reported) == 1
                else f"{len(not_reported)} results were not reported"
            )
        if unavailable:
            disposition_parts.append(
                "1 result had an unavailable source"
                if len(unavailable) == 1
                else f"{len(unavailable)} results had unavailable sources"
            )
        if conflicts:
            disposition_parts.append(
                "1 result retained a reference conflict"
                if len(conflicts) == 1
                else f"{len(conflicts)} results retained a reference conflict"
            )
        disposition_sentence = (
            "; ".join(disposition_parts) + "."
            if disposition_parts
            else ""
        )
        coverage_sentences = [
            f"{len(extracted)} of {len(outcome_cells)} Study results were available."
        ]
        if disposition_sentence:
            coverage_sentences.append(disposition_sentence)
        coverage_sentences.append(value_sentence)
        lines.append(
            f"**{label}.** {' '.join(coverage_sentences)}"
        )
        lines.append("")
    lines.extend(["### Published syntheses", ""])
    decisions = {item.analysis_id: item for item in decide_headlines(package)}
    for synthesis in _syntheses(package):
        certainty = _certainty_for_synthesis(package, synthesis)
        citations = _citations(synthesis)
        lines.append(
            f"**{_outcome_label(package, synthesis.outcome_id)}.** "
            f"{_headline_sentence(synthesis, decisions[synthesis.analysis_id])} "
            f"Independent certainty: {certainty.certainty.lower().replace('_', ' ')}; "
            f"source Review certainty: "
            f"{(certainty.source_review_certainty or 'NOT_REPORTED').lower().replace('_', ' ')}. "
            f"{citations}".rstrip()
        )
        lines.append("")
    return "\n".join(lines)


def _discussion_expansion(package, synthesis, decision) -> str:
    extracted = sum(
        fact.kind == "STUDY_OUTCOME" and fact.result_disposition == "EXTRACTED"
        for fact in package.facts
    )
    total = len(package.study_ids) * len(
        package.study_result_outcome_ids or package.outcome_ids
    )
    complete = package.fact_scope == "COMPLETE_SOURCE_SUPPORTED"
    risk_facts = [fact for fact in package.facts if fact.kind == "RISK_OF_BIAS"]
    risk_scope = (
        "the review-level evidence body"
        if risk_facts and risk_facts[0].assessment_level == "REVIEW_AGGREGATE"
        else "each included Study"
    )
    return (
        "### Interpretation of the evidence\n\n"
        f"The evidentiary record includes both the published syntheses and the Study-level result inventory: {extracted} of {total} prespecified Study by Outcome cells contained a complete extractable result. The remaining cells identify where the evidence was not reported, the source was unavailable, or a conflict prevented ordinary use. This coverage pattern matters for interpretation because a statistically precise synthesis for one outcome does not establish comprehensive benefit or safety across the review question.\n\n"
        "### Source conflicts and estimand validity\n\n"
        f"The reported synthesis validity disposition is {synthesis.validity}.\n\n"
        "### Risk of bias and certainty limitations\n\n"
        + (
            f"Source-bound risk-of-bias information was retained for {risk_scope} without expanding an aggregate judgment to unsupported Study-outcome claims. Outcome-specific certainty dispositions are shown in the Summary of Findings table.\n\n"
            if complete
            else "Result-specific risk-of-bias judgments were not reported for every Study and outcome. Outcome-specific certainty dispositions are shown in the Summary of Findings table.\n\n"
        )
        + "### Completeness of reporting\n\n"
        + (
            "The manuscript renders the source-bound inventory used for this analysis. Unsupported details, fields requiring named-author confirmation, and unresolved upstream source conflicts remain explicit.\n\n"
            if complete
            else "Complete identification and screening counts, full-text exclusion reasons, source-specific search dates or strategies, and some additional-analysis results were not reported in the available sources.\n\n"
        )
        + "### Applicability and future work\n\n"
        f"Applicability is limited to {package.scope.population}, the comparison of {package.scope.intervention} with {package.scope.comparator}, the prespecified outcomes, and their reported timepoints and analysis populations."
    )


def _interpretation_expansion(package) -> str:
    facts = [fact for fact in package.facts if fact.kind == "INTERPRETATION"]
    if not facts:
        return ""
    headings = {
        "PRINCIPAL_FINDING": "Source-bound principal interpretation",
        "APPLICABILITY": "Applicability",
        "LIMITATION": "Review limitations",
        "IMPLICATION": "Clinical and research implications",
        "COMPARISON_WITH_PRIOR_WORK": "Comparison with prior work",
    }
    return "\n\n".join(
        f"### {headings[fact.interpretation_kind]}\n\n"
        f"{_with_citations(fact, fact.statement_en)}"
        for fact in facts
    )


def _tables(package, decisions) -> str:
    studies = [fact for fact in package.facts if fact.kind == "STUDY"]
    robs = [fact for fact in package.facts if fact.kind == "RISK_OF_BIAS"]
    syntheses = {fact.outcome_id: fact for fact in _syntheses(package)}
    certainties = [fact for fact in package.facts if fact.kind == "CERTAINTY"]
    risk_title, _risk_caption = _risk_table_presentation(package)
    lines = [
        "# Tables",
        "## Table 1. Study characteristics",
        "| Study | Design | Sample size | Protocol population | Protocol intervention | Protocol comparator | Follow-up | Source-reported characteristics |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for fact in studies:
        lines.append(
            f"| {_cell(fact.study_id)} | {_cell(fact.design or 'Not reported')} | {fact.sample_size or 'Not reported'} | {_cell(fact.population or 'Not reported')} | {_cell(fact.intervention or 'Not reported')} | {_cell(fact.comparator or 'Not reported')} | {_cell(fact.follow_up or 'Not reported')} | {_cell(fact.source_characteristics or 'Not reported')} |"
        )
    lines.extend(
        [
            "",
            f"## Table 2. {risk_title}",
            "| Study | Outcome | Judgment | Rationale |",
            "|---|---|---|---|",
        ]
    )
    for fact in robs:
        lines.append(
            f"| {_cell(fact.study_id or 'All included studies')} | {_cell(fact.outcome_id or 'Review-level aggregate')} | {_cell(fact.judgment)} | {_cell(fact.rationale)} |"
        )
    lines.extend(
        [
            "",
            "## Table 3. Summary of findings",
            "| Outcome | Reported effect | Additional source-reported statistics | Synthesis validity | Headline disposition | Contributing Studies | Contributor identity | Independent certainty | Source Review certainty |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    decisions_by_analysis = {item.analysis_id: item for item in decisions}
    for certainty in certainties:
        synthesis = syntheses.get(certainty.outcome_id)
        lines.append(
            f"| {_cell(certainty.outcome_id)} | "
            f"{_cell(_effect_text(synthesis) if synthesis else 'Not reported')} | "
            f"{_cell(_additional_statistics(synthesis) if synthesis else 'Not reported')} | "
            f"{_cell(synthesis.validity if synthesis else 'NOT_APPLICABLE')} | "
            f"{_cell(decisions_by_analysis[synthesis.analysis_id].eligibility if synthesis else 'NOT_APPLICABLE')} | "
            f"{_cell(', '.join(synthesis.contributing_study_ids) if synthesis and synthesis.contributing_study_ids else str(synthesis.contributing_study_count) if synthesis and synthesis.contributing_study_count is not None else 'Not reported')} | "
            f"{_cell(synthesis.contributor_identity_disposition if synthesis else 'NOT_APPLICABLE')} | "
            f"{_cell(certainty.certainty)} | "
            f"{_cell(certainty.source_review_certainty or 'NOT_REPORTED')} |"
        )
    return "\n".join(lines) + "\n"


def _supplement(package, profile, decisions) -> str:
    lines = ["# Supplement", "## Search strategies and dates"]
    for fact in package.facts:
        if fact.kind == "SEARCH_SOURCE":
            lines.append(
                f"- **{fact.source_name}:** last search date: {fact.last_search_date or 'not reported'}; strategy: {fact.strategy or 'not reported in the available sources'}; disposition: {fact.strategy_disposition}."
            )
    lines.extend(["", "## Outcome definitions and timepoints"])
    for fact in package.facts:
        if fact.kind == "OUTCOME":
            lines.append(f"- **{fact.label}:** {fact.statement_en}")
    lines.extend(
        [
            "",
            "## Complete Study by Outcome inventory",
            "| Study | Outcome | Disposition | Values | Reason |",
            "|---|---|---|---|---|",
        ]
    )
    for fact in package.facts:
        if fact.kind != "STUDY_OUTCOME":
            continue
        lines.append(
            f"| {_cell(fact.study_id)} | {_cell(fact.outcome_id)} | {fact.result_disposition} | {_cell(_result_values(fact.values))} | {_cell(fact.reason or '')} |"
        )
    lines.extend(["", "## Additional analysis dispositions"])
    for fact in package.facts:
        if fact.kind == "ANALYSIS_AVAILABILITY":
            lines.append(f"- **{fact.analysis_kind}:** {fact.disposition}. {fact.statement_en}")
    lines.extend(["", "## PRISMA checklist dispositions"])
    for fact in package.facts:
        if fact.kind == "PRISMA_CHECKLIST_DISPOSITION":
            lines.append(
                f"- **{fact.guideline} item {fact.item_number} ({fact.item_label}):** {fact.status}. {fact.statement_en}"
            )
    lines.extend(["", "## Author responsibility dispositions"])
    for fact in package.facts:
        if fact.kind == "AUTHOR_RESPONSIBILITY":
            lines.append(f"- **{fact.field}:** {fact.statement_en}")
    if profile.supplement_contract.prisma_flow_placement == "SUPPLEMENT":
        lines.extend(
            [
                "",
                "## PRISMA flow",
                "The PRISMA flow diagram is presented in the supplement.",
            ]
        )
    return "\n".join(lines) + "\n"


def _references(package) -> str:
    lines = ["# References"]
    for index, reference in enumerate(package.references, start=1):
        lines.append(f"{index}. [{reference.citation_key}] {reference.display_text}")
    return "\n\n".join(lines) + "\n"


def _prisma_flow(package) -> str:
    fact = next(item for item in package.facts if item.kind == "SELECTION_FLOW")
    unknown = "Not reported"
    return (
        "# PRISMA 2020 flow\n\n"
        f"Records identified: {fact.records_identified if fact.records_identified is not None else unknown}\n\n"
        f"Duplicates removed: {fact.duplicates_removed if fact.duplicates_removed is not None else unknown}\n\n"
        f"Records screened: {fact.records_screened if fact.records_screened is not None else unknown}\n\n"
        f"Reports sought: {fact.reports_sought if fact.reports_sought is not None else unknown}\n\n"
        f"Reports not retrieved: {fact.reports_not_retrieved if fact.reports_not_retrieved is not None else unknown}\n\n"
        f"Reports assessed: {fact.reports_assessed if fact.reports_assessed is not None else unknown}\n\n"
        f"Reports excluded: {fact.reports_excluded if fact.reports_excluded is not None else unknown}\n\n"
        f"Studies included: {fact.studies_included}\n\n"
        f"Reports included: {fact.reports_included}\n\n"
        + (
            "Reported full-text exclusion reasons: "
            + "; ".join(
                f"{item.reason} ({item.report_count})" for item in fact.exclusion_reasons
            )
            + ".\n\n"
            if fact.exclusion_reasons
            else "Full-text exclusion reasons: Not reported in the bound sources.\n\n"
        )
        + "Unavailable counts are explicit dispositions and are not displayed as zero.\n"
    )


def _forest_svg(package, decisions) -> bytes:
    syntheses = _syntheses(package)
    decisions_by_analysis = {item.analysis_id: item for item in decisions}
    height = 70 + 58 * len(syntheses)
    body = [
        '<rect width="1000" height="100%" fill="white"/>',
        '<text x="30" y="28" font-size="18" font-weight="bold">Outcome synthesis inventory</text>',
    ]
    for index, synthesis in enumerate(syntheses):
        y = 65 + index * 58
        decision = decisions_by_analysis[synthesis.analysis_id]
        label = html.escape(_outcome_label(package, synthesis.outcome_id))
        effect = html.escape(_effect_text(synthesis))
        color = "#8b1e1e" if decision.eligibility == "AUDIT_ONLY" else "#16324f"
        body.extend(
            (
                f'<text data-outcome-id="{html.escape(synthesis.outcome_id)}" x="30" y="{y}" font-size="13" font-weight="bold">{label}</text>',
                f'<text x="30" y="{y + 20}" font-size="13" fill="{color}">{effect}</text>',
                f'<text x="770" y="{y + 20}" font-size="12" fill="{color}">{decision.eligibility}</text>',
            )
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">'
        + "".join(body)
        + "</svg>"
    ).encode()


def _lineage(package, profile) -> bytes:
    records = []
    for fact in package.facts:
        if fact.kind == "STUDY_OUTCOME":
            claim_text = (
                f"| {_cell(fact.study_id)} | {_cell(fact.outcome_id)} | {fact.result_disposition} | "
                f"{_cell(_result_values(fact.values))} | {_cell(fact.reason or '')} |"
            )
            artifact_paths = ["delivery/supplement.md"]
        elif fact.kind == "SYNTHESIS":
            if fact.validity == "NOT_REPORTED":
                claim_text = fact.outcome_id
                artifact_paths = [
                    "delivery/tables.md",
                    "delivery/figures/forest-plot.svg",
                ]
            else:
                claim_text = _effect_text(fact)
                artifact_paths = [
                    "delivery/manuscript.md",
                    "delivery/tables.md",
                    "delivery/figures/forest-plot.svg",
                ]
        elif fact.kind == "CERTAINTY":
            claim_text = f"| {fact.outcome_id} |"
            artifact_paths = ["delivery/tables.md"]
        elif fact.kind == "STUDY":
            claim_text = (
                f"| {_cell(fact.study_id)} | {_cell(fact.design or 'Not reported')} | "
                f"{fact.sample_size or 'Not reported'} |"
            )
            artifact_paths = ["delivery/tables.md"]
        elif fact.kind == "SELECTION_FLOW":
            claim_text = f"Studies included: {fact.studies_included}"
            artifact_paths = ["delivery/figures/prisma-flow.md"]
        elif fact.kind == "RISK_OF_BIAS":
            claim_text = (
                f"| {_cell(fact.study_id or 'All included studies')} | {_cell(fact.outcome_id or 'Review-level aggregate')} | "
                f"{_cell(fact.judgment)} | {_cell(fact.rationale)} |"
            )
            artifact_paths = ["delivery/tables.md"]
        elif fact.kind == "OUTCOME":
            claim_text = fact.statement_en
            artifact_paths = ["delivery/supplement.md"]
            if fact.statement_en in _results_expansion(
                package,
                compact=(
                    profile.word_budget.maximum is not None
                    and profile.word_budget.maximum <= 3000
                ),
            ):
                artifact_paths.append("delivery/manuscript.md")
        elif fact.kind == "REVIEW_METHOD":
            claim_text = fact.description
            artifact_paths = ["delivery/manuscript.md"]
        elif fact.kind == "INTERPRETATION":
            claim_text = fact.statement_en
            artifact_paths = ["delivery/manuscript.md"]
        elif fact.kind in {
            "ANALYSIS_AVAILABILITY",
            "AUTHOR_RESPONSIBILITY",
            "PRISMA_CHECKLIST_DISPOSITION",
        }:
            claim_text = fact.statement_en
            artifact_paths = ["delivery/supplement.md"]
        elif fact.kind == "SEARCH_SOURCE":
            claim_text = fact.statement_en
            artifact_paths = ["delivery/manuscript.md"]
        else:
            claim_text = (
                "systematic review"
                if getattr(fact, "topic", None) == "REVIEW_IDENTITY"
                else fact.statement_en
            )
            artifact_paths = ["delivery/manuscript.md"]
        if profile.profile_id == _CONTINUOUS_GENERIC_PROFILE_ID and fact.kind in {
            "STUDY",
            "SELECTION_FLOW",
            "RISK_OF_BIAS",
            "CERTAINTY",
        }:
            artifact_paths.append("delivery/manuscript.md")
        records.append(
            {
                "schema_version": "manuscript-claim-lineage.v2",
                "claim_id": f"claim-{len(records) + 1:05d}",
                "profile_id": profile.profile_id,
                "fact_ids": [fact.fact_id],
                "claim_text": claim_text,
                "artifact_paths": artifact_paths,
                "allowed_placements": list(fact.allowed_placements),
                "source_anchors": [
                    anchor.model_dump(mode="json") for anchor in fact.source_anchors
                ],
                "terminal_source_bindings": [
                    binding.model_dump(mode="json") for binding in fact.terminal_source_bindings
                ],
                "upstream_artifact_refs": list(fact.upstream_artifact_refs),
                "status": fact.status,
            }
        )
    return b"".join(canonical_json(record) + b"\n" for record in records)


def _narrative(package, topic):
    return next(
        fact.statement_en
        for fact in package.facts
        if fact.kind == "NARRATIVE" and fact.topic == topic
    )


def _risk_table_presentation(
    package: ManuscriptFactPackageV2,
) -> tuple[str, str]:
    robs = [fact for fact in package.facts if fact.kind == "RISK_OF_BIAS"]
    if robs and all(fact.assessment_level == "REVIEW_AGGREGATE" for fact in robs):
        return (
            "Review-level aggregate risk of bias",
            "Review-level aggregate risk-of-bias judgment and rationale",
        )
    if robs and all(fact.assessment_level == "STUDY" for fact in robs):
        return (
            "Study-level risk of bias",
            "Study-level risk-of-bias judgments and rationales",
        )
    return (
        "Result-specific risk of bias",
        "Result-specific risk-of-bias judgments and rationales",
    )


def _citations(fact) -> str:
    return " ".join(f"[{key}]" for key in fact.citation_keys)


def _with_citations(fact, text: str) -> str:
    citations = _citations(fact)
    return f"{text} {citations}".rstrip()


def _optional_narrative(package, topic: str, default: str) -> str:
    return next(
        (
            fact.statement_en
            for fact in package.facts
            if fact.kind == "NARRATIVE" and fact.topic == topic
        ),
        default,
    )


def _syntheses(package: ManuscriptFactPackageV2) -> tuple[SynthesisPublicationFact, ...]:
    by_outcome = {
        fact.outcome_id: fact for fact in package.facts if fact.kind == "SYNTHESIS"
    }
    return tuple(by_outcome[outcome_id] for outcome_id in package.outcome_ids if outcome_id in by_outcome)


def _primary_synthesis(package: ManuscriptFactPackageV2) -> SynthesisPublicationFact:
    syntheses = _syntheses(package)
    return next((fact for fact in syntheses if fact.is_primary), syntheses[0])


def _outcome_facts(package: ManuscriptFactPackageV2):
    return {fact.outcome_id: fact for fact in package.facts if fact.kind == "OUTCOME"}


def _outcome_label(package: ManuscriptFactPackageV2, outcome_id: str) -> str:
    fact = _outcome_facts(package).get(outcome_id)
    if fact is None:
        return outcome_id
    duplicate = sum(
        candidate.label == fact.label for candidate in _outcome_facts(package).values()
    ) > 1
    return f"{fact.label} ({fact.timepoint.rstrip('.')})" if duplicate else fact.label


def _outcome_labels(package: ManuscriptFactPackageV2) -> tuple[str, ...]:
    return tuple(_outcome_label(package, outcome_id) for outcome_id in package.outcome_ids)


def _study_label(study_id: str) -> str:
    return {
        "rescue_japan_limit": "RESCUE-Japan LIMIT",
        "angel_aspect": "ANGEL-ASPECT",
        "select2": "SELECT2",
        "tesla": "TESLA",
        "tension": "TENSION",
        "laste": "LASTE",
    }.get(study_id, study_id.replace("_", " ").title())


def _synthesis_summary(
    package: ManuscriptFactPackageV2,
    decisions: tuple[HeadlineDecision, ...],
    *,
    include_details: bool = True,
) -> str:
    by_analysis = {item.analysis_id: item for item in decisions}
    return " ".join(
        f"For {_outcome_label(package, fact.outcome_id)}, "
        + _decapitalize(
            _headline_sentence(fact, by_analysis[fact.analysis_id])
            if include_details
            else _compact_headline_sentence(fact, by_analysis[fact.analysis_id])
        )
        for fact in _syntheses(package)
    )


def _primary_synthesis_summary(
    package: ManuscriptFactPackageV2,
    decisions: tuple[HeadlineDecision, ...],
    *,
    include_details: bool = True,
) -> str:
    fact = _primary_synthesis(package)
    decision = next(item for item in decisions if item.analysis_id == fact.analysis_id)
    sentence = (
        _headline_sentence(fact, decision)
        if include_details
        else _compact_headline_sentence(fact, decision)
    )
    return f"For {_outcome_label(package, fact.outcome_id)}, {_decapitalize(sentence)}"


def _decapitalize(value: str) -> str:
    return value[:1].lower() + value[1:]


def _certainty_summary(package: ManuscriptFactPackageV2) -> str:
    def outcome_count(count: int) -> str:
        return f"{count} outcome" if count == 1 else f"{count} outcomes"

    certainties = [fact for fact in package.facts if fact.kind == "CERTAINTY"]
    grouped: dict[str, list[str]] = {}
    for fact in certainties:
        grouped.setdefault(fact.certainty.lower().replace("_", " "), []).append(
            _outcome_label(package, fact.outcome_id)
        )
    if len(certainties) > 3:
        independent = "Independent certainty was " + "; ".join(
            f"{certainty} for {outcome_count(len(labels))}"
            for certainty, labels in grouped.items()
        ) + "."
    else:
        independent = "Independent certainty was " + "; ".join(
            f"{fact.certainty.lower().replace('_', ' ')} for "
            f"{_outcome_label(package, fact.outcome_id)}"
            for fact in certainties
        ) + "."
    review_reported = [
        fact for fact in certainties if fact.source_review_certainty is not None
    ]
    return independent + (
        " The source Review reported outcome-specific certainty for "
        f"{outcome_count(len(review_reported))}."
        if review_reported
        else " The bound source Review exposed no exact outcome-specific certainty rating."
    )


def _risk_and_additional_analysis_summary(package: ManuscriptFactPackageV2) -> str:
    robs = [fact for fact in package.facts if fact.kind == "RISK_OF_BIAS"]
    analyses = [fact for fact in package.facts if fact.kind == "ANALYSIS_AVAILABILITY"]
    if package.fact_scope == "COMPLETE_SOURCE_SUPPORTED":
        risk_scope = (
            "the evidence body"
            if robs and robs[0].assessment_level == "REVIEW_AGGREGATE"
            else f"all {len(robs)} included Studies"
        )
        return (
            f"Source-bound risk-of-bias judgments were retained for {risk_scope}. "
            + " ".join(fact.statement_en for fact in analyses)
        )
    return (
        "The available evidence-body assessment could not be expanded into a source-bound result-specific judgment for every Study and outcome. These missing judgments are displayed rather than inferred. "
        "Sensitivity, subgroup, and reporting-bias analyses are likewise reported through explicit availability dispositions."
    )


def _limitation_suffix(package: ManuscriptFactPackageV2) -> str:
    if package.fact_scope == "COMPLETE_SOURCE_SUPPORTED":
        constraints = []
        if any(fact.status == "CONFLICT" for fact in package.facts):
            constraints.append("Unresolved upstream source conflicts remain explicit")
        if any(
            fact.kind == "AUTHOR_RESPONSIBILITY"
            and fact.status == "AWAITING_HUMAN_INPUT"
            for fact in package.facts
        ):
            constraints.append("fields requiring named-author confirmation remain open")
        return (
            f" {'; '.join(constraints)} and constrain finalization."
            if constraints
            else ""
        )
    return (
        " Missing search, selection, Study-characteristic, risk-of-bias, and author information "
        "limits interpretation and reproducibility."
    )


def _section_heading(profile: JournalPresentationProfile, section_id: str) -> str:
    return next(
        section.heading for section in profile.section_contract if section.section_id == section_id
    )


def _certainty_for_synthesis(package, synthesis: SynthesisPublicationFact):
    return next(
        fact
        for fact in package.facts
        if fact.kind == "CERTAINTY" and fact.outcome_id == synthesis.outcome_id
    )


def _headline_sentence(fact: SynthesisPublicationFact, decision: HeadlineDecision) -> str:
    if fact.validity == "NOT_REPORTED":
        return f"The bound sources reported {_effect_text(fact)}."
    details = _additional_statistics(fact, absent="")
    suffix = f"; {details}" if details else ""
    if decision.eligibility == "AUDIT_ONLY":
        if fact.validity == "VALID":
            return (
                f"The source review's valid pooled estimate was {_effect_text(fact)}{suffix}; "
                "contributing Study identity was incomplete, so headline promotion remains restricted."
            )
        reason = f"{fact.validity.lower().replace('_', ' ')} remains unresolved"
        return f"The source review's {_effect_text(fact)}{suffix} is retained for audit but is not presented as a valid pooled benefit or harm because {reason}."
    conflict = (
        " The recorded conflict remains visible."
        if decision.eligibility.endswith("CONFLICT")
        else ""
    )
    return f"The pooled result was {_effect_text(fact)}{suffix}.{conflict}"


def _compact_headline_sentence(
    fact: SynthesisPublicationFact, decision: HeadlineDecision
) -> str:
    if fact.validity == "NOT_REPORTED":
        return f"the bound sources reported {_effect_text(fact)}."
    if decision.eligibility == "AUDIT_ONLY":
        if fact.validity == "VALID":
            return (
                f"the source review reported the valid pooled estimate {_effect_text(fact)}, "
                "with incomplete contributing Study identity."
            )
        return (
            f"the source review's {_effect_text(fact)} is retained for audit because "
            f"{fact.validity.lower().replace('_', ' ')} remains unresolved."
        )
    return f"the pooled result was {_effect_text(fact)}."


def _effect_text(fact: SynthesisPublicationFact) -> str:
    if fact.estimate is None:
        return "no valid pooled estimate"
    heterogeneity = f"; {fact.heterogeneity}" if fact.heterogeneity else ""
    return f"{fact.effect_measure} {fact.estimate:g} (95% CI {fact.ci_lower:g} to {fact.ci_upper:g}{heterogeneity})"


def _additional_statistics(
    fact: SynthesisPublicationFact, *, absent: str = "Not reported"
) -> str:
    return "; ".join(
        value
        for value in (fact.p_value, fact.absolute_effect, fact.nnt)
        if value is not None
    ) or absent


def _result_values(values) -> str:
    if values is None:
        return "Not available"
    if values.result_type == "BINARY":
        return (
            f"{values.intervention_events}/{values.intervention_group_size} versus "
            f"{values.comparator_events}/{values.comparator_group_size}"
        )
    return (
        f"mean {values.intervention_mean:g} (SD {values.intervention_sd:g}), n={values.intervention_group_size} versus "
        f"mean {values.comparator_mean:g} (SD {values.comparator_sd:g}), n={values.comparator_group_size}"
    )


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


