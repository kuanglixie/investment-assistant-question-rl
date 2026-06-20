from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceGap:
    gap_id: str
    description: str
    severity: str = "medium"
    source_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchContext:
    ticker: str
    company_name: str | None = None
    thesis: str | None = None
    source_artifacts: tuple[str, ...] = ()
    evidence_gaps: tuple[EvidenceGap, ...] = ()
    existing_questions: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "thesis": self.thesis,
            "source_artifacts": list(self.source_artifacts),
            "evidence_gaps": [gap.to_dict() for gap in self.evidence_gaps],
            "existing_questions": list(self.existing_questions),
            "metrics": list(self.metrics),
        }


@dataclass(frozen=True)
class QuestionCandidate:
    question: str
    intent: str | None = None
    target_gap_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "target_gap_id": self.target_gap_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    label: str
    components: dict[str, float]
    penalties: dict[str, float]
    rationale: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "label": self.label,
            "components": self.components,
            "penalties": self.penalties,
            "rationale": list(self.rationale),
        }
