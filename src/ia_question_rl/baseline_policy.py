from __future__ import annotations

from ia_question_rl.models import QuestionCandidate, ResearchContext


def propose_questions(context: ResearchContext, max_questions: int = 5) -> list[QuestionCandidate]:
    """Generate simple source-seeking questions from evidence gaps."""

    candidates: list[QuestionCandidate] = []

    for gap in context.evidence_gaps:
        question = (
            f"Which official disclosures, source requests, or proxy metrics can test "
            f"the unresolved gap: {gap.description}?"
        )
        candidates.append(
            QuestionCandidate(
                question=question,
                intent="evidence_gap_follow_up",
                target_gap_id=gap.gap_id,
                metadata={"policy": "baseline_gap_template"},
            )
        )
        if len(candidates) >= max_questions:
            return candidates

    for metric in context.metrics:
        question = (
            f"What source-grounded evidence explains whether {metric} changes are durable, "
            f"temporary, or disclosure-driven for {context.ticker}?"
        )
        candidates.append(
            QuestionCandidate(
                question=question,
                intent="metric_anomaly_follow_up",
                metadata={"policy": "baseline_metric_template", "metric": metric},
            )
        )
        if len(candidates) >= max_questions:
            return candidates

    fallback = (
        f"What is the most material unanswered evidence gap for {context.ticker}, "
        "and which official source would resolve it?"
    )
    candidates.append(
        QuestionCandidate(
            question=fallback,
            intent="discover_evidence_gap",
            metadata={"policy": "baseline_fallback_template"},
        )
    )
    return candidates[:max_questions]
