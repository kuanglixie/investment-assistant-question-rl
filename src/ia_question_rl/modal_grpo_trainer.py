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
        "fastmcp",
        "python-dotenv",
        "httpx",
        "pypdf",
    )
    # Install hud-python directly from PyPI (matches user's uv environment)
    .pip_install("hud-python>=0.6.6")
    # Bake GRPO train and test prompt datasets directly into the container image
    .add_local_file("data/grpo_train_prompts.jsonl", remote_path="/root/grpo_train_prompts.jsonl")
    .add_local_file("data/grpo_test_prompts.jsonl", remote_path="/root/grpo_test_prompts.jsonl")
    # Bake HUD environment definitions directly into the container image
    .add_local_file("hud-investment/env.py", remote_path="/root/hud-investment/env.py")
    .add_local_file("hud-investment/tasks.py", remote_path="/root/hud-investment/tasks.py")
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
                f"The candidate research questions exhibit strong thematic alignment or target the same core business topic, financial trend, or strategic initiative as Golden Question #{i+1}: '{q}'",
                1.0,
            )
            for i, q in enumerate(targets)
        ]
        judge_question = (
            f"Evaluate whether the proposed research questions successfully cover the core business topics and strategic themes present in the golden analyst questions.\n"
            f"Evaluate each golden question independently. Award points (MET) if any candidate question targets the same overarching business topic, financial trend, or strategic initiative as the golden question, even if the candidate uses more rigorous forensic or accounting terminology.\n\n"
            f"=== GOLDEN ANALYST QUESTIONS ===\n{golden_questions_text}"
        )

        try:
            score, info = await LLMJudgeGrader.compute_score(
                answer=completion,
                criteria=criteria,
                question=judge_question,
                model="accounts/fireworks/models/glm-5p2",
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


def attach_tool_generate_wrapper(model: Any, tokenizer: Any) -> None:
    """Wrap model.generate to intercept tool calls and execute them against HUD Environment."""
    import asyncio
    import re
    import sys
    import threading
    import torch

    sys.path.insert(0, "/root/hud-investment")
    try:
        from env import env, _up, read_sec_document
        if not hasattr(env, "_bg_thread_started"):
            print("[INFO] Starting HUD Environment MCP server in background daemon thread...", flush=True)
            def run_server_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_up())
                loop.run_forever()

            t = threading.Thread(target=run_server_loop, daemon=True)
            t.start()
            env._bg_thread_started = True
    except Exception as e:
        print(f"[WARNING] Failed to initialize HUD environment in wrapper: {e}", flush=True)
        return

    original_generate = model.generate

    def generate_with_tools(*args: Any, **kwargs: Any) -> Any:
        if model.training:
            return original_generate(*args, **kwargs)

        outputs = original_generate(*args, **kwargs)
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return outputs

        prompt_len = input_ids.shape[1]
        gen_tokens = outputs[:, prompt_len:]
        decoded_list = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        new_outputs = []
        for i, text in enumerate(decoded_list):
            cur_tensor = outputs[i]
            if "read_sec_document" in text:
                print(f"[TOOL] Detected tool call in rollout {i}: {text[:150]}...", flush=True)
                match = re.search(r'read_sec_document\s*\(\s*[\'"]?([^\'"\)]+)[\'"]?\s*\)', text)
                target_path = match.group(1) if match else "annual_reports"

                try:
                    observation = asyncio.run(read_sec_document(target_path))
                except Exception as e:
                    observation = f"Tool execution error: {e}"

                print(f"[TOOL] Observation received (len={len(observation)}). Feeding back into model...", flush=True)
                tool_msg = f"\n<tool_response>{observation}</tool_response>\nGenerate final research questions:\n"
                tool_tokens = tokenizer.encode(tool_msg, return_tensors="pt").to(model.device)

                combined_input = torch.cat([cur_tensor.unsqueeze(0), tool_tokens], dim=1)
                turn2_kwargs = dict(kwargs)
                turn2_kwargs["input_ids"] = combined_input
                if "attention_mask" in turn2_kwargs:
                    turn2_kwargs["attention_mask"] = torch.ones_like(combined_input)

                final_output = original_generate(**turn2_kwargs)
                cur_tensor = final_output[0]

            new_outputs.append(cur_tensor)

        max_len = max(t.shape[0] for t in new_outputs)
        padded_outputs = []
        for t in new_outputs:
            if t.shape[0] < max_len:
                pad_tensor = torch.full((max_len - t.shape[0],), tokenizer.pad_token_id, dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad_tensor], dim=0)
            padded_outputs.append(t)

        return torch.stack(padded_outputs, dim=0)

    model.generate = generate_with_tools


@app.function(
    image=image,
    gpu="A100",  # Requires A100 or H100 to fit GRPOTrainer + LoRA weights + num_generations=8
    volumes={"/data": data_volume},
    timeout=21600,
    secrets=[
        modal.Secret.from_dict({
            "HUD_API_KEY": os.getenv("FIREWORKS_API_KEY", "fw_17S9CU1d1XMtJydFtTKQev"),
            "HUD_GATEWAY_URL": "https://api.fireworks.ai/inference/v1",
            "FIREWORKS_API_KEY": os.getenv("FIREWORKS_API_KEY", "fw_17S9CU1d1XMtJydFtTKQev"),
        }),
    ],
)
def train_grpo() -> None:
    """Main GRPO training execution entrypoint."""
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    print("[INFO] Initializing GRPO Training on Modal GPU container...", flush=True)

    # 1. Load GRPO train and test datasets from mounted files
    train_path = "/root/grpo_train_prompts.jsonl"
    test_path = "/root/grpo_test_prompts.jsonl"
    print(f"[INFO] Loading GRPO train dataset from {train_path}...", flush=True)
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    print(f"[INFO] Loading GRPO test dataset from {test_path}...", flush=True)
    eval_dataset = load_dataset("json", data_files=test_path, split="train")

    # 2. Initialize Model and Tokenizer (Qwen2.5-3B-Instruct guarantees powerful tool reasoning and zero gating errors)
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[INFO] Loading base model {model_id}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    # Attach HUD interactive tool rollout wrapper to model.generate
    attach_tool_generate_wrapper(model, tokenizer)

    # 3. Configure LoRA for parameter-efficient GRPO updates
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Define GRPO Hyperparameters (num_train_epochs=3 for full dataset alignment)
    # generation_batch_size=8 ensures perfect divisibility by num_generations=8 in TRL 1.6.0
    training_args = GRPOConfig(
        output_dir="/data/grpo_checkpoints",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        num_generations=8,  # G=8 candidate group size
        generation_batch_size=8,
        bf16=True,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )

    # 5. Define CustomGRPOTrainer for token loss masking on environment tool turns
    class CustomGRPOTrainer(GRPOTrainer):
        def _get_per_token_logps(self, model_obj: Any, input_ids: Any, attention_mask: Any, logits_to_keep: int) -> Any:
            per_token_logps = super()._get_per_token_logps(model_obj, input_ids, attention_mask, logits_to_keep)
            
            # Mask out <tool_response>...</tool_response> tokens so gradients only update on model turns
            tool_start_tokens = tokenizer.encode("\n<tool_response>", add_special_tokens=False)
            tool_end_tokens = tokenizer.encode("</tool_response>\nGenerate final research questions:\n", add_special_tokens=False)
            
            start_id = tool_start_tokens[0] if tool_start_tokens else None
            end_id = tool_end_tokens[-1] if tool_end_tokens else None
            
            if start_id is not None and end_id is not None:
                completion_ids = input_ids[:, -logits_to_keep:]
                for i in range(completion_ids.shape[0]):
                    seq = completion_ids[i]
                    in_tool = False
                    for j in range(seq.shape[0]):
                        if seq[j] == start_id:
                            in_tool = True
                        if in_tool:
                            per_token_logps[i, j] = 0.0 # Zero out logp to block gradient updates on environment turns
                        if seq[j] == end_id and in_tool:
                            in_tool = False
                            
            return per_token_logps

    # Initialize CustomGRPOTrainer with HUD LLMJudgeGrader reward function and eval dataset
    trainer = CustomGRPOTrainer(
        model=model,
        reward_funcs=[hud_judge_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_volume},
    timeout=3600,
    secrets=[
        modal.Secret.from_dict({
            "HUD_API_KEY": os.getenv("FIREWORKS_API_KEY", "fw_17S9CU1d1XMtJydFtTKQev"),
            "HUD_GATEWAY_URL": "https://api.fireworks.ai/inference/v1",
            "FIREWORKS_API_KEY": os.getenv("FIREWORKS_API_KEY", "fw_17S9CU1d1XMtJydFtTKQev"),
        }),
    ],
)
def evaluate_pre_and_post() -> None:
    """Evaluate pre-training vs post-training performance on test prompts using Fireworks AI LLMJudgeGrader."""
    import json
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    test_path = "/root/grpo_test_prompts.jsonl"
    print(f"[EVAL] Loading test prompts from {test_path}...", flush=True)
    
    prompts = []
    golden_questions = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data["prompt"])
            golden_questions.append(data["golden_questions"])

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[EVAL] Loading base model {model_id} (pre-training)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    attach_tool_generate_wrapper(base_model, tokenizer)

    def generate_completions(model: Any) -> list[str]:
        completions = []
        for prompt_messages in prompts:
            text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            completions.append(completion)
        return completions

    print("[EVAL] Generating completions for pre-training model...", flush=True)
    pre_completions = generate_completions(base_model)
    
    print("[EVAL] Scoring pre-training completions with HUD LLMJudgeGrader...", flush=True)
    dummy_prompts = [""] * len(pre_completions)
    pre_scores = hud_judge_reward_func(dummy_prompts, pre_completions, golden_questions)
    pre_mean = sum(pre_scores) / len(pre_scores) if pre_scores else 0.0
    print(f"[EVAL] Pre-training Mean Reward: {pre_mean:.4f}", flush=True)

    final_dir = "/data/grpo_checkpoints/final_lora"
    print(f"[EVAL] Loading post-training LoRA model from {final_dir}...", flush=True)
    peft_model = PeftModel.from_pretrained(base_model, final_dir)
    attach_tool_generate_wrapper(peft_model, tokenizer)
    
    print("[EVAL] Generating completions for post-training model...", flush=True)
    post_completions = generate_completions(peft_model)
    
    print("[EVAL] Scoring post-training completions with HUD LLMJudgeGrader...", flush=True)
    post_scores = hud_judge_reward_func(dummy_prompts, post_completions, golden_questions)
    post_mean = sum(post_scores) / len(post_scores) if post_scores else 0.0
    print(f"[EVAL] Post-training Mean Reward: {post_mean:.4f}", flush=True)

    print("\n" + "="*50, flush=True)
    print(f"=== FINAL EVALUATION METRICS ===", flush=True)
    print(f"Pre-training Mean Reward:  {pre_mean:.4f}", flush=True)
    print(f"Post-training Mean Reward: {post_mean:.4f}", flush=True)
    print("="*50 + "\n", flush=True)


@app.local_entrypoint()
def main() -> None:
    """CLI entrypoint for `modal run src/ia_question_rl/modal_grpo_trainer.py`."""
    print("[INFO] Submitting GRPO training job to Modal container network...", flush=True)
    train_grpo.remote()
    print("[INFO] Running evaluation for pre-training vs post-training...", flush=True)
    evaluate_pre_and_post.remote()
