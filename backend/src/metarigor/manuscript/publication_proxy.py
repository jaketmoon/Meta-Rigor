from __future__ import annotations

"""从封存 reference 形成独立写作事实边界；原包和来源始终保留。"""


import json
from .publication_models import ManuscriptFactPackageV2, publication_fact_sha256
POLICY = "ms-reference-upstream-proxy-v1"


OBJECTIVES = {
    "balbaa-2025": (
        "Adults undergoing percutaneous cardiac or cardiac device procedures; "
        "non-fasting versus conventional fasting; patient comfort and safety."
    ),
    "lin-2025": (
        "Adults with acute large-core ischemic stroke; endovascular thrombectomy plus "
        "medical care versus medical care alone; efficacy and safety."
    ),
    "malhotra-2025": (
        "Adults with active cancer after at least six months of anticoagulation for "
        "venous thromboembolism; reduced-dose versus full-dose apixaban; efficacy and "
        "safety."
    ),
    "meng-2025": (
        "Adults undergoing cardiac surgery; intravenous dexmedetomidine versus placebo "
        "or normal saline; postoperative delirium, ICU length of stay, mortality, "
        "bradycardia and hypotension."
    ),
    "qazi-2026": (
        "Adults with imaging-selected acute ischemic stroke, 4.5 to 24 hours after last"
        " known well; intravenous tenecteplase versus randomized control treatment; "
        "efficacy and safety."
    ),
    "song-2025": (
        "Adults aged 18 years or older with asymptomatic severe aortic stenosis; early "
        "aortic valve replacement versus conservative management; efficacy outcomes."
    ),
    "sun-2024": (
        "Adults undergoing cardiac surgery with cardiopulmonary bypass; near-infrared "
        "spectroscopy-guided monitoring versus control care; postoperative delirium and"
        " secondary outcomes."
    ),
    "turalde-mapili-2023": (
        "Infants no older than one year; single intramuscular dose of nirsevimab versus"
        " placebo; RSV efficacy and safety."
    ),
    "wang-2025": (
        "Adults with sepsis or septic shock; intravenous-fluid resuscitation guided by "
        "dynamic measures of fluid responsiveness versus no dynamic guidance; "
        "patient-important clinical and resuscitative outcomes."
    ),
    "zhong-2022": (
        "Children with bronchiolitis; high-flow nasal cannula versus non-invasive "
        "positive-pressure ventilation, including CPAP and NPPV; treatment failure, "
        "intubation and PICU length of stay."
    ),
}


FIELDS = {
    "NARRATIVE": ("topic",),
    "OUTCOME": (
        "outcome_id",
        "label",
        "definition",
        "timepoint",
        "role",
        "study_result_disposition",
    ),
    "REVIEW_METHOD": ("method_domain", "method_basis", "description"),
    "SEARCH_SOURCE": (
        "source_name",
        "source_type",
        "last_search_date",
        "strategy",
        "strategy_disposition",
    ),
    "SELECTION_FLOW": (
        "records_identified",
        "duplicates_removed",
        "records_screened",
        "reports_sought",
        "reports_not_retrieved",
        "reports_assessed",
        "reports_excluded",
        "studies_included",
        "reports_included",
        "exclusion_reasons",
    ),
    "STUDY": (
        "study_id",
        "report_ids",
        "design",
        "population",
        "intervention",
        "comparator",
        "sample_size",
        "follow_up",
        "source_characteristics",
    ),
    "STUDY_OUTCOME": ("study_id", "outcome_id", "result_disposition", "values", "reason"),
    "SYNTHESIS": (
        "analysis_id",
        "outcome_id",
        "effect_measure",
        "estimate",
        "ci_lower",
        "ci_upper",
        "heterogeneity",
        "p_value",
        "absolute_effect",
        "nnt",
        "is_primary",
        "contributing_study_ids",
        "contributing_study_count",
        "contributor_identity_disposition",
        "analysis_population",
        "timepoint",
        "validity",
    ),
    "ANALYSIS_AVAILABILITY": ("analysis_kind", "outcome_id", "disposition"),
    "RISK_OF_BIAS": ("study_id", "assessment_level", "outcome_id", "judgment", "rationale"),
    "CERTAINTY": (
        "outcome_id",
        "certainty",
        "rationale",
        "source_review_certainty",
        "source_review_certainty_rationale",
    ),
    "AUTHOR_RESPONSIBILITY": ("field", "value", "provided_by"),
    "PRISMA_CHECKLIST_DISPOSITION": ("guideline", "item_number", "item_label"),
}


ANALYSIS_DETAILS = {
    "pub-complete-lin-analysis-1": (
        "Imaging subgroup estimates: non-contrast CT or CT perfusion, RR 2.62 (95% CI "
        "1.76 to 3.91); MRI, RR 2.39 (95% CI 1.34 to 4.27)."
    ),
    "pub-complete-lin-analysis-2": (
        "Serial leave-one-out analysis was reported; source-reported diagnostic: pooled"
        " estimates similar to overall estimates."
    ),
    "pub-complete-lin-analysis-3": (
        "Primary-outcome funnel plot: source-reported visual diagnostic, no major asymmetry."
    ),
    "pub-complete-song-2025-analysis-detail-1": (
        "Primary-composite random-effects 95% prediction interval: 0.18 to 1.59. "
        "Excluding RECOVERY: I²=0%. Source-reported leave-one-out diagnostic: no "
        "material change in estimate."
    ),
    "pub-complete-song-2025-analysis-detail-2": (
        "Intervention-strategy subgroup analysis: source-reported diagnostic, no "
        "substantial change in primary-composite effect."
    ),
    "pub-complete-song-2025-analysis-detail-3": (
        "Funnel plots: source-reported visual diagnostic, no obvious publication bias."
    ),
    "pub-complete-wang-2025-analysis-detail-1": (
        "High-versus-low risk-of-bias subgroup analyses covered ICU mortality, ICU "
        "length of stay, hospital length of stay, and mechanical ventilation; "
        "source-reported diagnostic: no credible subgroup effect."
    ),
}


def typed_fields(fact) -> dict:
    raw = fact.model_dump(mode="json")
    result = {key: raw[key] for key in FIELDS[fact.kind]}
    if fact.kind == "NARRATIVE":
        result["scope_description"] = fact.statement_en
    elif fact.kind == "ANALYSIS_AVAILABILITY":
        result["reported_details"] = fact.statement_en
    return result


def build_proxy(source: ManuscriptFactPackageV2) -> tuple[ManuscriptFactPackageV2, dict]:
    if source.fact_scope != "COMPLETE_SOURCE_SUPPORTED":
        raise ValueError("Reference proxy requires an explicit complete source package")
    facts, audit = [], []
    for fact in source.facts:
        row = {"fact_id": fact.fact_id, "old_sha256": fact.content_sha256}
        if fact.kind == "INTERPRETATION" or (
            fact.kind == "NARRATIVE" and fact.topic in {"LIMITATION", "REGISTRATION"}
        ):
            audit.append(
                {
                    **row,
                    "disposition": "EXCLUDED",
                    "reason": "REFERENCE_INTERPRETATION_OR_ORIGINAL_REVIEW_REGISTRATION",
                    "new_sha256": None,
                }
            )
            continue
        value = fact.model_dump(mode="json")
        fields = {key: value[key] for key in FIELDS[fact.kind]}
        if fact.kind == "NARRATIVE":
            statement = {
                "REVIEW_IDENTITY": (
                    "Systematic review and pairwise meta-analysis of intervention studies."
                ),
                "OBJECTIVE": OBJECTIVES[source.case_id],
                "CONTEXT": OBJECTIVES[source.case_id],
                "ELIGIBILITY": (
                    "Eligibility is specified in the accepted protocol; final-review methods are "
                    "identified separately."
                ),
            }.get(fact.topic)
            if statement is None:
                raise ValueError(f"Uncurated narrative topic: {fact.topic}")
        elif fact.kind == "ANALYSIS_AVAILABILITY":
            statement = ANALYSIS_DETAILS.get(fact.fact_id, fact.statement_en)
        else:
            statement = f"status: {fact.status}; " + "; ".join(
                f"{key}: "
                f"{json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else val}"
                for key, val in fields.items()
                if val is not None
            )
        value["statement_en"] = statement
        value["content_sha256"] = publication_fact_sha256(value)
        normalized = type(fact).model_validate(value)
        facts.append(normalized)
        audit.append(
            {
                **row,
                "disposition": "NORMALIZED",
                "reason": "TYPED_FIELDS_AND_EXPLICIT_RESEARCH_RESULT_CURATION",
                "new_sha256": normalized.content_sha256,
                "retained_fields": list(fields),
            }
        )
    slots = {slot for fact in facts for slot in fact.required_for}
    payload = source.model_dump(mode="json")
    payload.update(
        package_id=f"{source.package_id}-llm-proxy-v1",
        fact_scope="REFERENCE_UPSTREAM_PROXY",
        facts=[fact.model_dump(mode="json") for fact in facts],
        required_slots=[slot for slot in source.required_slots if slot in slots],
    )
    payload["curation"].update(
        curator=POLICY,
        curated_at="2026-09-10T00:00:00Z",
        input_sha256s=[*source.curation.input_sha256s, source.canonical_sha256],
        notes=(
            "Reference upstream proxy for MS-only evaluation. Source-review interpretation "
            "and original registration excluded by input policy, not marked NOT_REPORTED. "
            "No new search, extraction, synthesis, or author declarations executed."
        ),
    )
    package = ManuscriptFactPackageV2.model_validate(payload)
    return package, {
        "schema_version": "manuscript-proxy-curation.v1",
        "policy": POLICY,
        "case_id": source.case_id,
        "source_package_sha256": source.canonical_sha256,
        "proxy_package_sha256": package.canonical_sha256,
        "facts": audit,
        "limitations": [
            "REFERENCE_UPSTREAM_NOT_LIVE_SPECIALIST_CHAIN",
            "NO_INDEPENDENT_BACKGROUND_OR_PRIOR_WORK_INPUT",
        ],
    }


