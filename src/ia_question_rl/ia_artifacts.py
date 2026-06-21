from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ia_question_rl.models import EvidenceGap, ResearchContext


KNOWN_ARTIFACTS = {
    "financial_report_pack.json",
    "layer1_question_pack.json",
    "evidence_communication_pack.json",
    "feedback_loop_pack.json",
    "source_map.json",
    "artifact_contracts.json",
    "state.json",
}


def discover_artifacts(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"Run directory does not exist: {root}")
    found: dict[str, Path] = {}
    for path in root.rglob("*.json"):
        if path.name in KNOWN_ARTIFACTS and path.name not in found:
            found[path.name] = path
    return found


def context_from_run(
    run_dir: str | Path,
    ticker: str,
    thesis: str | None = None,
    company_name: str | None = None,
) -> ResearchContext:
    artifacts = discover_artifacts(run_dir)
    payloads = {name: _load_json(path) for name, path in artifacts.items()}

    existing_questions = _dedupe(_collect_strings_by_key(payloads, {"question", "research_question"}))
    metrics = _dedupe(_collect_strings_by_key(payloads, {"metric", "metric_name", "metric_family"}))
    gaps = _extract_gaps(payloads)

    return ResearchContext(
        ticker=ticker,
        company_name=company_name,
        thesis=thesis,
        source_artifacts=tuple(str(path) for path in artifacts.values()),
        evidence_gaps=tuple(gaps),
        existing_questions=tuple(existing_questions),
        target_human_questions=(),
        metrics=tuple(metrics),
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_strings_by_key(payload: Any, keys: set[str]) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and isinstance(value, str):
                values.append(value)
            else:
                values.extend(_collect_strings_by_key(value, keys))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_collect_strings_by_key(item, keys))
    return values


def _extract_gaps(payloads: dict[str, Any]) -> list[EvidenceGap]:
    candidates: list[EvidenceGap] = []
    for artifact_name, payload in payloads.items():
        for item in _walk_dicts(payload):
            description = _first_string(
                item,
                (
                    "gap",
                    "gap_description",
                    "description",
                    "request",
                    "follow_up",
                    "open_question",
                    "missing_item",
                ),
            )
            if not description:
                continue
            if not _looks_like_gap(item, description):
                continue
            gap_id = _first_string(item, ("gap_id", "id", "question_id", "request_id")) or _slug(description)
            severity = _first_string(item, ("severity", "priority", "materiality")) or "medium"
            candidates.append(
                EvidenceGap(
                    gap_id=gap_id,
                    description=description,
                    severity=severity,
                    source_refs=(artifact_name,),
                )
            )
    return _dedupe_gaps(candidates)


def _walk_dicts(payload: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        items.append(payload)
        for value in payload.values():
            items.extend(_walk_dicts(value))
    elif isinstance(payload, list):
        for value in payload:
            items.extend(_walk_dicts(value))
    return items


def _first_string(item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _looks_like_gap(item: dict[str, Any], description: str) -> bool:
    key_text = " ".join(str(key).lower() for key in item.keys())
    description_text = description.lower()
    markers = (
        "gap",
        "missing",
        "unresolved",
        "follow",
        "request",
        "open_question",
        "unknown",
        "human_review",
    )
    return any(marker in key_text or marker in description_text for marker in markers)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = " ".join(value.split()).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(value)
    return output


def _dedupe_gaps(gaps: list[EvidenceGap]) -> list[EvidenceGap]:
    seen: set[str] = set()
    output: list[EvidenceGap] = []
    for gap in gaps:
        key = gap.gap_id.lower()
        if key not in seen:
            seen.add(key)
            output.append(gap)
    return output


def _slug(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text)
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts[:8]) or "gap"
