# Integration With InvestmentAssistant

This repo is designed as a sidecar to the current InvestmentAssistant research stack.

## Expected Inputs

Known useful artifacts include:

- `financial_report_pack.json`
- `layer1_question_pack.json`
- `evidence_communication_pack.json`
- `feedback_loop_pack.json`
- `source_map.json`
- `artifact_contracts.json`
- `state.json`

The first adapter is deliberately tolerant: it recursively searches a run directory for known filenames and extracts question-like, gap-like, and metric-like fields without requiring a perfect schema match.

## Boundary

This repo should not become another research agent. Its job is narrower:

1. Read research state.
2. Propose better next questions.
3. Score/rank candidate questions.
4. Produce auditable training/eval episodes.

The main InvestmentAssistant repo can later import a stable policy or consume exported question candidates, but this repo should first prove the loop outside the production path.

## Suggested Main-Repo Hook Later

After the offline harness is validated, a low-risk integration point is:

```text
feedback_loop_pack + layer1_question_pack
        |
        v
question_rl_policy.rank_candidates(...)
        |
        v
top_k follow-up questions stored as question candidates
```

The policy should write candidates with:

- `question`;
- `target_gap_id`;
- `required_sources`;
- `reward_breakdown`;
- `policy_version`;
- `episode_id`.

It should not invent facts, change source state, or silently trigger collection.
