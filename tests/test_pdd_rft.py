import json
from pathlib import Path

from eval_protocol import EvaluationRow, evaluation_test


@evaluation_test(input_dataset=["data/dpo_pdd_training_pairs.jsonl"])
def test_pdd_question_generation(row: EvaluationRow) -> EvaluationRow:
    """Run Reinforcement Fine-Tuning (DPO) on GLM 5.2 using PDD SEC exhibits and gold questions."""
    task_file = Path("data/episodes/pdd_supervised_dpo_task.json")
    if not task_file.exists():
        row.evaluation_result = 0.0
        return row

    with task_file.open("r", encoding="utf-8") as f:
        task_data = json.load(f)
    gold_questions = task_data.get("gold_questions", [])

    if not gold_questions or not row.messages:
        row.evaluation_result = 0.0
        return row

    # Extract model generated content from the final assistant message
    generated_content = row.messages[-1].content or ""
    if not generated_content:
        row.evaluation_result = 0.0
        return row

    # Split generated content into separate questions/paragraphs
    generated_questions = [q.strip() for q in generated_content.split("\n") if q.strip() and "?" in q]
    if not generated_questions:
        generated_questions = [generated_content]

    match_count = 0
    for gold_q in gold_questions:
        gold_tokens = set(gold_q.lower().split())
        if not gold_tokens:
            continue
        for gen_q in generated_questions:
            gen_tokens = set(gen_q.lower().split())
            if not gen_tokens:
                continue
            overlap = len(gold_tokens.intersection(gen_tokens)) / len(gold_tokens.union(gen_tokens))
            if overlap >= 0.2:  # Consider it a match if token overlap >= 20%
                match_count += 1
                break

    # Evaluation result is the ratio of gold questions successfully matched
    row.evaluation_result = min(1.0, match_count / len(gold_questions))
    return row
