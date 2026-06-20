from __future__ import annotations

import re
from collections.abc import Iterable

from ia_question_rl.models import ResearchContext, RewardBreakdown


SOURCE_TERMS = {
    "10-k",
    "20-f",
    "6-k",
    "filing",
    "disclosure",
    "official",
    "segment",
    "cohort",
    "unit economics",
    "margin",
    "revenue",
    "cash flow",
    "source",
    "evidence",
    "proxy",
    "competitor",
    "pricing",
}

DECISION_TERMS = {
    "durability",
    "valuation",
    "risk",
    "margin",
    "growth",
    "cash",
    "profitability",
    "thesis",
    "variant",
    "unit economics",
    "market share",
}

VAGUE_PATTERNS = (
    re.compile(r"\bwhat do you think\b", re.I),
    re.compile(r"\bis (it|this|that) good\b", re.I),
    re.compile(r"\bwhat is going on\b", re.I),
    re.compile(r"\btell me about\b", re.I),
)


def evaluate_question(question: str, context: ResearchContext) -> RewardBreakdown:
    """Score a candidate question with an inspectable first-pass rubric."""

    normalized = _normalize(question)
    tokens = set(_tokens(normalized))
    rationale: list[str] = []

    components = {
        "materiality": _materiality(tokens, context),
        "answerability": _answerability(normalized),
        "evidence_gap_fit": _evidence_gap_fit(tokens, context),
        "novelty": _novelty(tokens, context.existing_questions),
        "source_grounding": _source_grounding(normalized),
        "decision_relevance": _decision_relevance(normalized, context),
        "specificity": _specificity(tokens, normalized, context),
    }

    penalties = {
        "vagueness": _vagueness_penalty(normalized, tokens),
        "overbreadth": _overbreadth_penalty(normalized),
        "conclusion_first": _conclusion_first_penalty(normalized),
    }

    if components["evidence_gap_fit"] >= 1.5:
        rationale.append("Targets a known evidence gap.")
    if components["source_grounding"] >= 1.0:
        rationale.append("Names evidence or source types.")
    if components["novelty"] <= 0.25:
        rationale.append("Likely duplicates an existing question.")
    if penalties["vagueness"] > 0:
        rationale.append("Too vague for a research workpaper.")

    raw_total = sum(components.values()) - sum(penalties.values())
    total = max(0.0, round(raw_total, 2))
    label = _label(total)
    return RewardBreakdown(
        total=total,
        label=label,
        components={key: round(value, 2) for key, value in components.items()},
        penalties={key: round(value, 2) for key, value in penalties.items()},
        rationale=tuple(rationale),
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower())


def _token_overlap(tokens: set[str], texts: Iterable[str]) -> int:
    pool: set[str] = set()
    for text in texts:
        pool.update(_tokens(text))
    return len(tokens & pool)


def _materiality(tokens: set[str], context: ResearchContext) -> float:
    thesis_text = context.thesis or ""
    gap_texts = [gap.description for gap in context.evidence_gaps]
    metric_texts = list(context.metrics)
    overlap = _token_overlap(tokens, [thesis_text, *gap_texts, *metric_texts])
    if overlap >= 5:
        return 2.0
    if overlap >= 2:
        return 1.25
    if tokens & DECISION_TERMS:
        return 0.75
    return 0.0


def _answerability(text: str) -> float:
    score = 0.0
    if text.endswith("?") or text.startswith(("what ", "which ", "how ", "why ", "where ", "whether ")):
        score += 0.5
    if any(term in text for term in SOURCE_TERMS):
        score += 1.0
    if any(term in text for term in ("test", "verify", "confirm", "measure", "isolate", "compare")):
        score += 0.75
    return min(score, 2.0)


def _evidence_gap_fit(tokens: set[str], context: ResearchContext) -> float:
    if not context.evidence_gaps:
        return 0.0
    best_overlap = max(_jaccard(tokens, set(_tokens(gap.description))) for gap in context.evidence_gaps)
    if best_overlap >= 0.25:
        return 2.0
    if best_overlap >= 0.12:
        return 1.25
    return 0.0


def _novelty(tokens: set[str], existing_questions: Iterable[str]) -> float:
    existing = list(existing_questions)
    if not existing:
        return 1.0
    max_overlap = max(_jaccard(tokens, set(_tokens(question))) for question in existing)
    if max_overlap >= 0.8:
        return 0.0
    if max_overlap >= 0.55:
        return 0.35
    return 1.0


def _source_grounding(text: str) -> float:
    hits = sum(1 for term in SOURCE_TERMS if term in text)
    if hits >= 2:
        return 1.5
    if hits == 1:
        return 0.75
    return 0.0


def _decision_relevance(text: str, context: ResearchContext) -> float:
    hits = sum(1 for term in DECISION_TERMS if term in text)
    if context.thesis and _jaccard(set(_tokens(text)), set(_tokens(context.thesis))) >= 0.15:
        hits += 1
    if hits >= 2:
        return 1.5
    if hits == 1:
        return 0.75
    return 0.0


def _specificity(tokens: set[str], text: str, context: ResearchContext) -> float:
    score = 0.0
    if context.ticker and context.ticker.lower() in text:
        score += 0.5
    if context.company_name and context.company_name.lower() in text:
        score += 0.5
    if len(tokens) >= 10:
        score += 0.5
    if any(char.isdigit() for char in text):
        score += 0.25
    if any(term in text for term in ("standalone", "separate", "by segment", "versus", "rather than")):
        score += 0.75
    return min(score, 1.5)


def _vagueness_penalty(text: str, tokens: set[str]) -> float:
    penalty = 0.0
    if len(tokens) < 6:
        penalty += 1.0
    if any(pattern.search(text) for pattern in VAGUE_PATTERNS):
        penalty += 1.0
    return penalty


def _overbreadth_penalty(text: str) -> float:
    question_marks = text.count("?")
    conjunctions = len(re.findall(r"\b(and|or|plus|also)\b", text))
    if question_marks > 1 or conjunctions >= 4:
        return 0.5
    return 0.0


def _conclusion_first_penalty(text: str) -> float:
    if re.search(r"\bshould we (buy|sell|short|avoid)\b", text):
        return 0.75
    return 0.0


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _label(total: float) -> str:
    if total >= 8.0:
        return "excellent"
    if total >= 5.0:
        return "useful"
    return "weak"
