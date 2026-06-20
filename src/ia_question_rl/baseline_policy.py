from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from urllib.error import HTTPError, URLError

from ia_question_rl.models import QuestionCandidate, ResearchContext


def propose_questions(context: ResearchContext, max_questions: int = 5) -> list[QuestionCandidate]:
    """Generate candidate research questions via Jetski CLI or Fireworks AI API, falling back to baseline templates."""
    if os.environ.get("USE_JETSKI_CLI") == "1":
        try:
            return _propose_questions_jetski(context, max_questions)
        except Exception as e:
            print(f"[Jetski CLI Policy Error] {e}. Falling back to baseline template policy.")

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if api_key:
        try:
            return _propose_questions_fireworks(context, api_key, max_questions)
        except Exception as e:
            print(f"[Fireworks AI Policy Error] {e}. Falling back to baseline template policy.")

    return _propose_questions_baseline(context, max_questions)


def _propose_questions_jetski(context: ResearchContext, max_questions: int) -> list[QuestionCandidate]:
    model = os.environ.get("JETSKI_MODEL", "pro")

    system_prompt = """You are an expert investment research assistant generating high-quality follow-up research questions.
Your goal is to propose specific, material, answerable research questions targeting known evidence gaps.

Rules for high-quality questions:
1. Grounding: Explicitly name source types (e.g., '10-K', '20-F', '6-K', 'segment disclosures', 'unit economics', 'filings').
2. Answerability: Use testable action verbs ('test', 'verify', 'confirm', 'measure', 'isolate', 'compare').
3. Specificity: Include the company ticker and target specific evidence gaps or metric anomalies.
4. No Conclusions: Never ask subjective conclusion questions ('should we buy/sell/short/avoid').
5. Concise & Singular: Do not chain multiple questions together with excessive conjunctions (and, or, plus).

Return ONLY a JSON array of objects, where each object has:
- "question": The candidate research question string.
- "intent": "evidence_gap_follow_up" or "metric_anomaly_follow_up".
- "target_gap_id": The ID of the evidence gap being targeted (if applicable)."""

    gap_summaries = [
        {"gap_id": gap.gap_id, "description": gap.description}
        for gap in context.evidence_gaps[:max_questions]
    ]

    user_prompt = f"""Generate up to {max_questions} research questions for ticker {context.ticker}.
Thesis: {context.thesis or 'N/A'}
Evidence Gaps: {json.dumps(gap_summaries, ensure_ascii=False)}
Known Metrics: {list(context.metrics[:10])}

Output ONLY the JSON array of objects. Do not include markdown code blocks or explanatory text."""

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    cmd = ["agentapi", "new-conversation", f"--model={model}", full_prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    content = result.stdout.strip()

    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    items = json.loads(content.strip())
    candidates: list[QuestionCandidate] = []
    for item in items:
        candidates.append(
            QuestionCandidate(
                question=item["question"],
                intent=item.get("intent", "evidence_gap_follow_up"),
                target_gap_id=item.get("target_gap_id"),
                metadata={"policy": "jetski_cli", "model": model},
            )
        )
    return candidates[:max_questions]


def _propose_questions_fireworks(context: ResearchContext, api_key: str, max_questions: int) -> list[QuestionCandidate]:
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    model = os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/deepseek-r1-distill-qwen-7b")

    system_prompt = """You are an expert investment research assistant generating high-quality follow-up research questions.
Your goal is to propose specific, material, answerable research questions targeting known evidence gaps.

Rules for high-quality questions:
1. Grounding: Explicitly name source types (e.g., '10-K', '20-F', '6-K', 'segment disclosures', 'unit economics', 'filings').
2. Answerability: Use testable action verbs ('test', 'verify', 'confirm', 'measure', 'isolate', 'compare').
3. Specificity: Include the company ticker and target specific evidence gaps or metric anomalies.
4. No Conclusions: Never ask subjective conclusion questions ('should we buy/sell/short/avoid').
5. Concise & Singular: Do not chain multiple questions together with excessive conjunctions (and, or, plus).

Return ONLY a JSON array of objects, where each object has:
- "question": The candidate research question string.
- "intent": "evidence_gap_follow_up" or "metric_anomaly_follow_up".
- "target_gap_id": The ID of the evidence gap being targeted (if applicable)."""

    gap_summaries = [
        {"gap_id": gap.gap_id, "description": gap.description}
        for gap in context.evidence_gaps[:max_questions]
    ]

    user_prompt = f"""Generate up to {max_questions} research questions for ticker {context.ticker}.
Thesis: {context.thesis or 'N/A'}
Evidence Gaps: {json.dumps(gap_summaries, ensure_ascii=False)}
Known Metrics: {list(context.metrics[:10])}

Output ONLY the JSON array of objects. Do not include markdown code blocks or explanatory text."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    content = response_data["choices"][0]["message"]["content"].strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    items = json.loads(content.strip())
    candidates: list[QuestionCandidate] = []
    for item in items:
        candidates.append(
            QuestionCandidate(
                question=item["question"],
                intent=item.get("intent", "evidence_gap_follow_up"),
                target_gap_id=item.get("target_gap_id"),
                metadata={"policy": "fireworks_ai", "model": model},
            )
        )
    return candidates[:max_questions]


def _propose_questions_baseline(context: ResearchContext, max_questions: int = 5) -> list[QuestionCandidate]:
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
