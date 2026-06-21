# Architecture

This project treats research-question generation as a policy-learning problem around InvestmentAssistant artifacts.

## Design Principle

The policy should learn from the workpaper state, not from a blank chat prompt.

Investment research questions are only useful when they are connected to:

- a thesis or decision point;
- an evidence gap;
- available or missing sources;
- prior questions already asked;
- the next phase of the research workflow.

## Loop

```text
InvestmentAssistant run artifacts
        |
        v
Artifact adapter
        |
        v
Episode(context, existing_questions, gaps, metrics)
        |
        v
Question policy -> candidate questions
        |
        v
Reward model / rubric / human preference
        |
        v
Training set, eval set, policy update
```

## Observation

The observation is the compact state a question policy is allowed to see:

- `ticker` and optional `company_name`;
- current research `thesis`;
- known source artifacts;
- metric names or anomaly labels;
- evidence gaps from `feedback_loop_pack` or equivalent;
- existing research questions from `layer1_question_pack` or draft metadata.

## Action

An action is one candidate question:

```json
{
  "question": "Which official disclosures can test whether Temu unit economics are improving without relying on management narrative?",
  "intent": "evidence_gap_follow_up",
  "target_gap_id": "temu_standalone_economics"
}
```

## Reward

The initial reward is intentionally inspectable. It scores:

- materiality;
- answerability;
- evidence-gap fit;
- novelty versus existing questions;
- source grounding;
- decision relevance.

It penalizes:

- vague prompts;
- duplicated questions;
- questions that cannot be answered by available or requestable evidence;
- questions that ask for conclusion before evidence.

## Why Offline First

For this project, offline RL/evaluation is the safe first step. The system can use historical runs and manually reviewed question pairs before any online loop touches the main research workflow.

Possible later steps:

- preference dataset from analyst A/B choices;
- reward model trained on pairwise question comparisons;
- GRPO-style policy tuning;
- constrained online bandit that only proposes follow-up questions, not conclusions.
