# Hackathon Resources & Provider Mapping

This document catalogs the available hackathon credits and outlines how each provider can be utilized to accelerate, scale, and train the reinforcement learning and evidence collection pipelines for `InvestmentAssistant`.

## Available Hackathon Credits

| Provider | Credit Amount | Code / Details | Sign-up / Redemption Link |
| :--- | :--- | :--- | :--- |
| **Fireworks** | $500 | `HUD-HACK` | See instructions |
| **Modal** | $250 | `SQ8-USG-5K2` | [modal.com/credits](https://modal.com/credits) |
| **HUD** | $200 | Unique code emailed | Emailed to hackathon sign-up email |
| **Antim** | $150 | Pre-loaded upon sign-up | [gizmo.antimlabs.com/](https://gizmo.antimlabs.com/) |
| **Daytona** | $100 | `DAYTONA_RL_ENVIRONMENTS_HACK_Y6ZDQBG5` | [app.daytona.io/dashboard/billing](https://app.daytona.io/dashboard/billing) |
| **Exa** | $50 | `HUDHACK` | [dashboard.exa.ai/billing](https://dashboard.exa.ai/billing) |
| **MiniMax** | $30 | Form registration | [Feishu Form](https://vrfi1sk8a0.feishu.cn/share/base/form/shrcnn25RnUEWP5VlAQ6JyOwwZc) |
| **DeepMind** | $25 | GCP credits pre-loaded | Create GCP account using hackathon email |
| **Anthropic** | $25 | Offer code | [Claude Offers](https://claude.com/offers?offer_code=1b352ac9-2a3e-428d-9694-104178861a2e) |
| **Sixtyfour** | 64 credits | Pre-loaded upon sign-up | [sixtyfour.ai](https://www.sixtyfour.ai/) |

---

## High-Impact Architectural Mapping

### 1. Fireworks AI: RL Training & High-Speed Rollouts ($500 Credit)
Fireworks AI provides both blazing-fast inference and managed fine-tuning services, making it the ideal engine for our RL loop:
* **RL / DPO Fine-Tuning**: Fireworks supports fine-tuning open-source base models (e.g., Llama 3, Mistral) via LoRA. Once our offline evaluation harness generates scored episode datasets matching `episode.schema.json`, we can convert them into preference pairs (chosen vs. rejected questions) and launch DPO (Direct Preference Optimization) training jobs directly on Fireworks infrastructure.
* **Massive Candidate Rollouts**: During policy iteration, we can query Fireworks at high concurrency to generate batches of question candidates in `baseline_policy.py` at a fraction of the cost and latency of larger proprietary models.
* **Instant LoRA Serving**: Once the RL policy is fine-tuned, Fireworks instantly serves the LoRA adapter on top of base models with zero cold-start penalty.

### 2. Modal ($250) & Daytona ($100): Distributed Compute & RL Environments
* **Modal**: Excellent for containerized serverless execution. We can deploy the offline evaluation harness across dozens of parallel Modal workers to parse historical SEC runs, evaluate candidates against the `evaluate_question` rubric in `reward.py`, and aggregate large-scale training episodes instantly.
* **Daytona**: Offers standardized, reproducible cloud development environments. Perfect for hosting long-running RL environment loops and managing workspace state during the hackathon.

### 3. Exa ($50): Neural Web Search & Alternative Evidence
* **Web Grounding**: While `investment-assistant-core` relies on SEC EDGAR filings (`sec_edgar.py`), many `EvidenceGap` items require alternative data, competitor pricing, or industry commentary. Exa's neural search can be integrated as a complementary collector to fetch high-quality web evidence to answer unresolved gaps.

### 4. Anthropic ($25) & MiniMax ($30): Advanced Judge & Synthesis Models
* **LLM-as-a-Judge**: Use Claude 3.5 Sonnet / MiniMax as advanced evaluator models to periodically validate or calibrate our rule-based `evaluate_question` rubric, ensuring our inspectable reward function aligns with expert human analyst reasoning.

### 5. DeepMind/GCP ($25), HUD ($200), Antim ($150), Sixtyfour (64 credits): Infrastructure & Storage
* **Data Persistence**: Store large caches of raw SEC filings (`data/raw/sec`), extracted report packs, and training episode JSONL tables securely in GCP buckets or specialized agent hosting platforms.
