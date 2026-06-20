# InvestmentAssistant Question RL

这个 repo 是 `InvestmentAssistant` 的旁路 RL 实验项目：目标不是先改主系统，而是先把“什么是好的 research question”变成可记录、可评分、可训练、可回放的对象。

核心问题：

> How can InvestmentAssistant ask the right research question at the right time?

这里的 “right” 暂时定义为：

- 对投资判断有 material impact；
- 能被官方披露、工作底稿、竞品/替代数据或人工 follow-up 回答；
- 针对真实 evidence gap，而不是泛泛提问；
- 与已有问题不重复；
- 能推进 `collect -> workpaper -> reconcile -> judge -> memo` 的下一步。

## Why A Separate Repo

主 `InvestmentAssistant` 已经会产出结构化研究产物，例如 `financial_report_pack.json`、`layer1_question_pack.json`、`evidence_communication_pack.json`、`feedback_loop_pack.json` 和最终 research draft。这个 repo 先消费这些产物，训练/评估一个 question policy，再决定哪些能力值得回写主仓库。

这样可以避免在主 pipeline 里直接堆 agent，也让 RL 实验有清楚的输入、动作、奖励和回放证据。

## First Runnable Loop

```bash
PYTHONPATH=src python3 -m ia_question_rl.cli score \
  --ticker PDD \
  --thesis "Temu margin durability and cash conversion" \
  --gap "Temu standalone economics are not disclosed separately" \
  --question "Which official disclosures or segment proxies can test whether Temu unit economics are improving without relying on management narrative?"
```

从 InvestmentAssistant run 目录抽取 context 并生成 baseline candidates：

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
schemas/
  episode.schema.json
tests/
```

## RL Framing

- Observation: company/ticker, thesis, existing questions, known metrics, source artifacts, evidence gaps.
- Action: a candidate research question plus metadata.
- Reward: materiality, answerability, evidence-gap fit, novelty, source grounding, and decision relevance, minus vagueness/redundancy penalties.
- Episode: one InvestmentAssistant research state or report pack snapshot, plus generated candidates and reward feedback.

The first milestone is not PPO. The first milestone is a reliable offline evaluation harness. Once the reward rubric agrees with human review on enough examples, the repo can support preference modeling, DPO/RLAIF-style ranking, or online policy iteration.
