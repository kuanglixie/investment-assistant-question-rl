# Research Question Rubric

The reward rubric answers one practical question:

> Would this question help an analyst decide what to inspect next?

## Dimensions

### Materiality

High-scoring questions affect the investment thesis, valuation, risk, durability, or variant perception.

Weak:

- "What is going on with the company?"

Strong:

- "Which disclosures can test whether margin expansion came from durable mix shift rather than temporary marketing pullback?"

### Answerability

Good questions point toward evidence that can be collected, cited, or explicitly marked missing.

Signals:

- official filings;
- earnings releases or investor presentations;
- segment metrics;
- cohort/market/competitor data;
- source requests for unavailable evidence.

### Evidence-Gap Fit

The best question targets a known uncertainty rather than generating a generic checklist.

Example:

- Gap: `temu_standalone_economics`
- Good question: "Which official disclosures or proxy metrics can isolate Temu unit economics from the domestic marketplace?"

### Novelty

A question loses value if it restates an existing question with different wording.

The first baseline uses token overlap and exact normalization. Later versions should use embeddings or human preference labels.

### Source Grounding

A strong question names the type of source that could answer it or defines a source request.

### Decision Relevance

A question should move the next research phase:

- collect;
- workpaper;
- reconcile;
- judge;
- memo.

## Label Guide

- `excellent`: specific, material, answerable, gap-linked, non-duplicative.
- `useful`: directionally good but missing some specificity.
- `weak`: vague, redundant, or only loosely tied to the thesis.
