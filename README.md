# InvestmentAssistant Question RL

This repository is a side-car reinforcement learning (RL) experimental project for `InvestmentAssistant`. Instead of modifying the primary system immediately, the goal is to turn "what makes a good research question" into an object that can be recorded, scored, trained, and replayed.

Core Question:

> How can InvestmentAssistant ask the right research question at the right time?

Here, "right" is currently defined as:

- Has a material impact on investment judgments;
- Can be answered via official disclosures, workpapers, competitor/alternative data, or manual follow-ups;
- Targets a genuine evidence gap rather than asking generic questions;
- Does not duplicate existing questions;
- Drives the next step in the `collect -> workpaper -> reconcile -> judge -> memo` workflow.

## Why A Separate Repo

The main `InvestmentAssistant` already outputs structured research artifacts, such as `financial_report_pack.json`, `layer1_question_pack.json`, `evidence_communication_pack.json`, `feedback_loop_pack.json`, and the final research draft. This repository first consumes these artifacts to train and evaluate a question policy before deciding which capabilities are worth merging back into the main repository.

This approach avoids cluttering the main pipeline with agents and ensures the RL experiments have clear inputs, actions, rewards, and replayable evidence.

## First Runnable Loop

```bash
PYTHONPATH=src python3 -m ia_question_rl.cli score \
  --ticker PDD \
  --thesis "Temu margin durability and cash conversion" \
  --gap "Temu standalone economics are not disclosed separately" \
  --question "Which official disclosures or segment proxies can test whether Temu unit economics are improving without relying on management narrative?"
```

Extract context from an InvestmentAssistant run directory and generate baseline candidates:

```bash
PYTHONPATH=src python3 -m ia_question_rl.cli extract-episode \
  --run-dir /path/to/investment-assistant/run \
  --ticker PDD \
  --thesis "Assess whether recent growth is durable and cash-generative" \
  --output data/episodes/pdd.jsonl
```

## Repo Layout

```text
src/ia_question_rl/
  models.py              # Observation/action/reward dataclasses.
  reward.py              # Inspectable first reward baseline.
  baseline_policy.py     # Simple policy that proposes questions from evidence gaps.
  ia_artifacts.py        # Adapter for InvestmentAssistant artifact folders.
  cli.py                 # score and extract-episode commands.
docs/
  architecture.md
  integration-with-investment-assistant.md
  research-question-rubric.md
  hackathon-resources.md
schemas/
  episode.schema.json
tests/
```

## RL Framing

- Observation: company/ticker, thesis, existing questions, known metrics, source artifacts, evidence gaps.
- Action: a candidate research question plus metadata.
- Reward: materiality, answerability, evidence-gap fit, novelty, source grounding, and decision relevance, minus vagueness/redundancy penalties.
- Episode: one InvestmentAssistant research state or report pack snapshot, plus generated candidates and reward feedback.

The first milestone is not PPO. The first milestone is a reliable offline evaluation harness. Once the reward rubric agrees with human review on enough examples, the repo can support preference modeling, GRPO/RLAIF-style ranking, or online policy iteration.
