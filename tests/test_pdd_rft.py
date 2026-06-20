from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval_protocol import RFTTask, evaluation_test


@evaluation_test(model="accounts/fireworks/models/glm-5p2")
def test_pdd_question_generation() -> RFTTask:
    """Run Reinforcement Fine-Tuning (RFT/DPO) on GLM 5.2 using PDD SEC exhibits and gold questions."""
    task_file = Path("data/episodes/pdd_supervised_dpo_task.json")
    with task_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return RFTTask(
        prompt=data["task_prompt"],
        attachment=data["attachment"],
        gold_targets=data["gold_questions"],
        rft_config={"n_epochs": 3, "beta": 0.1, "learning_rate_multiplier": 1.0},
    )
