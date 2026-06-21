"""Modal GRPO Trainer for InvestmentAssistant Question RL.

Runs HuggingFace TRL GRPOTrainer inside a Modal GPU container, utilizing HUD's
LLMJudgeGrader (via HUD Gateway or Fireworks AI) to evaluate candidate questions online.

Usage:
    modal run src/ia_question_rl/modal_grpo_trainer.py
"""

import asyncio
import os
import sys
from typing import Any, cast

import modal

# Define Modal environment image with all necessary ML and HUD dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "trl>=0.14.0",
        "datasets",
        "accelerate",
        "peft",
        "pydantic",
        "pydantic-settings",
        "openai",
    )
    # Install hud-python directly from PyPI (matches user's uv environment)
    .pip_install("hud-python>=0.6.6")
    # Bake GRPO prompts dataset directly into the container image
    .add_local_file("data/grpo_pdd_training_prompts.jsonl", remote_path="/root/grpo_pdd_training_prompts.jsonl")
)

app = modal.App("ia-question-grpo")
data_volume = modal.Volume.from_name("ia-finance-data", create_if_missing=True)


def hud_judge_reward_func(prompts: list[str], completions: list[str], golden_questions: list[list[str]], **kwargs: Any) -> list[float]:
    """GRPO Reward Function wrapper. Computes M/N golden coverage using HUD LLMJudgeGrader."""
    # Import HUD inside the container worker
    from hud.graders import LLMJudgeGrader

    async def _score_single(completion: str, targets: list[str]) -> float:
        if not targets:
            return 0.0

        golden_questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(targets))
        criteria = [
            (
                f"The candidate research questions exhibit strong thematic alignment or target the same core business topic (e.g., international expansion/Temu, margin trends, investment initiatives, or macro consumer backdrop) as Golden Question #{i+1}: '{q}'",
                1.0,
            )
            for i, q in enumerate(targets)
        ]
        judge_question = (
            f"Evaluate whether the proposed research questions successfully cover the core business topics and strategic themes present in the golden analyst questions.\n"
            f"Evaluate each golden question independently. Award points (MET) if any candidate question targets the same overarching business topic, financial trend, or strategic initiative (such as international expansion, profit margin fluctuations, or new investment initiatives) as the golden question, even if the candidate uses more rigorous forensic or accounting terminology.\n\n"
            f"=== GOLDEN ANALYST QUESTIONS ===\n{golden_questions_text}"
        )

        try:
            score, info = await LLMJudgeGrader.compute_score(
                answer=completion,
                criteria=criteria,
                question=judge_question,
                model="claude-haiku-4-5",  # Or accounts/fireworks/models/llama-v3p1-70b-instruct
            )
            # Round score to exact M / N representation
            total_golden = len(targets)
            covered = int(round(score * total_golden))
            return float(covered) / float(total_golden) if total_golden > 0 else 0.0
        except Exception as e:
            print(f"[ERROR] LLMJudgeGrader failed during GRPO rollout: {e}", file=sys.stderr)
            return 0.0

    async def _score_all() -> list[float]:
        tasks = [_score_single(comp, gold) for comp, gold in zip(completions, golden_questions)]
        return list(await asyncio.gather(*tasks))

    # Execute async judging loop synchronously for GRPOTrainer
    return asyncio.run(_score_all())


@app.function(
    image=image,
    gpu="A100",  # Requires A100 or H100 to fit GRPOTrainer + LoRA weights + num_generations=8
    volumes={"/data": data_volume},
    timeout=3600,
    secrets=[
        modal.Secret.from_dict({"HUD_API_KEY": os.getenv("HUD_API_KEY", "sk-hud-7Cq9BWsUlQ-oA25R2Es9s-3ewfIUFOnM-ds")}),
    ],
)
def train_grpo() -> None:
    """Main GRPO training execution entrypoint."""
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    print("[INFO] Initializing GRPO Training on Modal GPU container...", flush=True)

    # 1. Load GRPO prompts dataset from mounted file and isolate single example for goal verification
    dataset_path = "/root/grpo_pdd_training_prompts.jsonl"
    print(f"[INFO] Loading GRPO dataset from {dataset_path} and selecting single example...", flush=True)
    dataset = load_dataset("json", data_files=dataset_path, split="train").select([0])

    # 2. Initialize Model and Tokenizer (Qwen2.5-0.5B-Instruct guarantees ultra-fast download and zero gating errors)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[INFO] Loading base model {model_id}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    # 3. Configure LoRA for parameter-efficient GRPO updates
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Define GRPO Hyperparameters (max_steps=1 to execute exactly one successful single-example rollout loop)
    # generation_batch_size=8 ensures perfect divisibility by num_generations=8 in TRL 1.6.0
    training_args = GRPOConfig(
        output_dir="/data/grpo_checkpoints",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        num_generations=8,  # G=8 candidate group size
        generation_batch_size=8,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    # 5. Initialize GRPOTrainer with HUD LLMJudgeGrader reward function
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[hud_judge_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    print("[INFO] Starting GRPOTrainer online rollout loop...", flush=True)
    trainer.train()

    # 6. Save final LoRA weights to mounted volume
    final_dir = "/data/grpo_checkpoints/final_lora"
    print(f"[INFO] Training complete! Saving final LoRA checkpoint to {final_dir}...", flush=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("[INFO] Modal GRPO training run finished successfully!", flush=True)


@app.local_entrypoint()
def main() -> None:
    """CLI entrypoint for `modal run src/ia_question_rl/modal_grpo_trainer.py`."""
    print("[INFO] Submitting GRPO training job to Modal container network...", flush=True)
    train_grpo.remote()
