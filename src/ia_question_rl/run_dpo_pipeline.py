from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from ia_question_rl.baseline_policy import propose_questions
from ia_question_rl.models import EvidenceGap, ResearchContext
from ia_question_rl.reward import evaluate_question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end DPO data generation and Fireworks AI RL pipeline.")
    parser.add_argument("--task-file", default="data/episodes/pdd_supervised_dpo_task.json")
    parser.add_argument("--output-dpo", default="data/dpo_pdd_training_pairs.jsonl")
    parser.add_argument("--submit-fireworks", action="store_true", help="Submit DPO fine-tuning job to Fireworks AI.")
    args = parser.parse_args(argv)

    task_path = Path(args.task_file)
    if not task_path.exists():
        print(f"[Error] Task file not found: {task_path}")
        return 1

    with task_path.open("r", encoding="utf-8") as handle:
        task_data = json.load(handle)

    task_prompt = task_data["task_prompt"]
    attachment_dir = Path(task_data["attachment"])
    gold_questions = task_data["gold_questions"]

    print(f"[1/5] Loaded task data from {task_path}. Gold questions count: {len(gold_questions)}")

    # Ingest local attachment files
    attachment_summary = ""
    if attachment_dir.exists():
        files = list(attachment_dir.rglob("*"))
        print(f"[2/5] Ingested local attachment directory {attachment_dir}. Found {len(files)} files.")
        attachment_summary = f"Ingested {len(files)} local SEC filing exhibits from {attachment_dir}."
    else:
        print(f"[2/5] Warning: Attachment directory {attachment_dir} not found.")

    # Construct ResearchContext for policy generation and reward scoring
    context = ResearchContext(
        ticker="PDD",
        thesis="Assess whether recent growth is durable and cash-generative",
        target_human_questions=tuple(gold_questions),
        evidence_gaps=tuple(
            [
                EvidenceGap(gap_id="temu_growth", description="Temu standalone economics are not disclosed separately."),
                EvidenceGap(gap_id="margin_fluctuation", description="Fluctuation in cost-to-profit ratio and gross margin pressure."),
                EvidenceGap(gap_id="first_party_brand", description="RMB 100 billion investment plan for first-party brand initiative."),
            ]
        ),
    )

    print("[3/5] Generating policy rollouts via configured AI provider (Fireworks / Jetski / Baseline)...")
    candidates = propose_questions(context, max_questions=len(gold_questions))

    # Evaluate candidates with reward rubric
    scored_candidates = []
    for cand in candidates:
        reward = evaluate_question(cand.question, context)
        scored_candidates.append((cand.question, reward.total, reward.label))

    print(f"[4/5] Evaluated {len(scored_candidates)} candidate rollouts against gold_questions ground truth.")

    # Construct DPO Preference Pairs
    dpo_output_path = Path(args.output_dpo)
    dpo_output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    with dpo_output_path.open("w", encoding="utf-8") as handle:
        for index, gold_q in enumerate(gold_questions):
            rejected_q = scored_candidates[index % len(scored_candidates)][0] if scored_candidates else "What is going on with the company?"
            dpo_pair = {
                "messages": [
                    {"role": "system", "content": task_prompt},
                    {
                        "role": "user",
                        "content": f"Analyze the attached SEC documents and generate expert research questions.\n\nAttachment Summary: {attachment_summary}",
                    },
                ],
                "chosen": {"role": "assistant", "content": gold_q},
                "rejected": {"role": "assistant", "content": rejected_q},
            }
            handle.write(json.dumps(dpo_pair, ensure_ascii=False) + "\n")
            pairs.append(dpo_pair)

    print(f"[5/5] Successfully generated {len(pairs)} DPO preference pairs. Wrote to {dpo_output_path}.")

    if args.submit_fireworks:
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            print("[Fireworks AI Error] FIREWORKS_API_KEY environment variable is required to submit DPO fine-tuning job.")
            return 1
        _submit_fireworks_dpo_job(api_key, str(dpo_output_path))

    return 0


def _submit_fireworks_dpo_job(api_key: str, training_file: str) -> None:
    url = "https://api.fireworks.ai/v1/fine_tuning/jobs"
    print(f"[Fireworks AI] Submitting DPO fine-tuning job using training file {training_file}...")

    payload = {
        "model": "accounts/fireworks/models/glm-5p2",
        "training_file": training_file,
        "hyperparameters": {
            "n_epochs": 3,
            "learning_rate_multiplier": 1.0,
            "loss_type": "dpo",
            "beta": 0.1,
        },
        "suffix": "ia-question-rl-pdd-dpo",
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            print(f"[Fireworks AI Success] Fine-tuning job submitted successfully! Job ID: {res_data.get('id')}")
    except Exception as e:
        print(f"[Fireworks AI Submit Error] Failed to submit fine-tuning job: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
