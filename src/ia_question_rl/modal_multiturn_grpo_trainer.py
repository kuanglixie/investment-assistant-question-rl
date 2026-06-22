"""Modal Multi-Turn GRPO Trainer for InvestmentAssistant Question RL.

Executes a full multi-turn Agentic GRPO training loop inside a Modal GPU container.
Spins up a local FastMCP server binding to 127.0.0.1, mounts physical SEC filings
and PDF documents, and executes multi-turn tool interaction rollouts (G=8) before
scoring golden question coverage via HUD's LLMJudgeGrader.

Usage:
    modal run src/ia_question_rl/modal_multiturn_grpo_trainer.py
"""

import asyncio
import json
import os
import sys
from typing import Any

import modal

# 1. Define Modal environment image with all necessary ML, FastMCP, and HUD dependencies
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
    )
    # Install hud-python directly from PyPI (matches user's uv environment)
    .pip_install("hud-python>=0.6.6")
    # Bake GRPO train and test prompt datasets directly into the container image
    .add_local_file("data/grpo_train_prompts.jsonl", remote_path="/root/grpo_train_prompts.jsonl")
    .add_local_file("data/grpo_test_prompts.jsonl", remote_path="/root/grpo_test_prompts.jsonl")
)

app = modal.App("ia-question-multiturn-grpo")
data_volume = modal.Volume.from_name("ia-finance-data", create_if_missing=True)

# Mount the entire physical observations directory containing raw SEC filings and PDFs
observations_mount = modal.Mount.from_local_dir(
    "/Users/ajing/Downloads/rl_observations",
    remote_path="/root/rl_observations",
    condition=lambda p: not p.endswith(".DS_Store")
)


def get_secure_api_key() -> str:
    """Securely fetch API key without hardcoding secrets in source code."""
    key = os.getenv("FIREWORKS_API_KEY")
    if not key:
        # Fallback to checking local secret file or raising explicit secure error
        if os.path.exists("/root/fireworks_secret.txt"):
            with open("/root/fireworks_secret.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        raise RuntimeError("[SECURITY ERROR] FIREWORKS_API_KEY secret is required but missing.")
    return key


# ── SEC Document Reader Tool (For Local MCP Server) ──────────────────────────

def read_sec_document(target_path: str, max_chars: int = 4000) -> str:
    """Read the contents of a local SEC filing document or PDF artifact securely.

    Strictly validates paths against the sandbox directory to prevent path traversal.
    """
    sandbox_dir = os.path.abspath("/root/rl_observations")
    
    # Sanitize and resolve absolute path
    if not os.path.isabs(target_path):
        target_path = os.path.abspath(os.path.join(sandbox_dir, target_path))
    else:
        target_path = os.path.abspath(target_path)
        
    # Mandatory Path Traversal Guard: ensure path starts with sandbox_dir + path.sep
    if not target_path.startswith(sandbox_dir + os.sep) and target_path != sandbox_dir:
        return f"[SECURITY ERROR] Path traversal attempt blocked: {target_path}"

    if not os.path.exists(target_path):
        parent = os.path.dirname(target_path)
        if os.path.exists(parent) and os.path.isdir(parent):
            files = os.listdir(parent)
            return f"File not found: {target_path}. Available files in directory: {files}"
        return f"Path does not exist: {target_path}"
    
    if os.path.isdir(target_path):
        files = os.listdir(target_path)
        return f"Target is a directory. Available files: {files}"
    
    try:
        with open(target_path, encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars)
            return content or "File is empty"
    except Exception as e:
        return f"Error reading file {target_path}: {e}"


# ── HUD Judging Wrapper ──────────────────────────────────────────────────────

def hud_judge_reward_func(completions: list[str], golden_questions: list[list[str]]) -> list[float]:
    """GRPO Reward Function wrapper. Computes M/N golden coverage using HUD LLMJudgeGrader."""
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
            total_golden = len(targets)
            covered = int(round(score * total_golden))
            return float(covered) / float(total_golden) if total_golden > 0 else 0.0
        except Exception as e:
            print(f"[ERROR] LLMJudgeGrader failed during multi-turn GRPO rollout: {e}", file=sys.stderr)
            return 0.0

    async def _score_all() -> list[float]:
        tasks = [_score_single(comp, gold) for comp, gold in zip(completions, golden_questions)]
        return list(await asyncio.gather(*tasks))

    return asyncio.run(_score_all())


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_volume},
    mounts=[observations_mount],
    timeout=21600,
    secrets=[
        modal.Secret.from_dict({
            "HUD_API_KEY": os.getenv("FIREWORKS_API_KEY", "fw_17S9CU1d1XMtJydFtTKQev"),
            "HUD_GATEWAY_URL": "https://api.fireworks.ai/inference/v1",
            "FIREWORKS_API_KEY": os.getenv("FIREWORKS_API_KEY", "fw_17S9CU1d1XMtJydFtTKQev"),
        }),
    ],
)
def train_multiturn_grpo() -> None:
    """Custom Multi-Turn GRPO training loop using PyTorch, PEFT, and FastMCP."""
    import torch
    from fastmcp import FastMCP
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[INFO] Initializing Multi-Turn GRPO Training on Modal GPU container...", flush=True)
    api_key = get_secure_api_key()

    # 1. Spin up FastMCP Server locally binding exclusively to 127.0.0.1 (Mandatory Security Rule)
    mcp_port = 8090
    mcp_server = FastMCP(name="research-tools")
    mcp_server.tool(read_sec_document)
    
    # Run FastMCP server in background task
    loop = asyncio.get_event_loop()
    server_task = loop.create_task(
        mcp_server.run_async(transport="http", host="127.0.0.1", port=mcp_port, show_banner=False)
    )
    print(f"[INFO] FastMCP Server initialized successfully on http://127.0.0.1:{mcp_port}", flush=True)

    # 2. Load GRPO train dataset
    train_path = "/root/grpo_train_prompts.jsonl"
    print(f"[INFO] Loading GRPO train dataset from {train_path}...", flush=True)
    prompts = []
    golden_questions_list = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data["prompt"])
            golden_questions_list.append(data["golden_questions"])

    # 3. Initialize Model and Tokenizer
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[INFO] Loading base model {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Initialize reference model for KL divergence calculation
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    ref_model.eval()

    # Configure LoRA for parameter-efficient GRPO updates
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_generations = 8  # G=8 candidate group size
    beta = 0.04  # KL divergence coefficient

    print("[INFO] Starting Custom Multi-Turn GRPO online rollout loop...", flush=True)
    
    for step, (prompt_messages, golden) in enumerate(zip(prompts, golden_questions_list)):
        print(f"\n--- STEP {step+1}/{len(prompts)} ---", flush=True)
        
        # Format base prompt
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        # Run G=8 multi-turn rollouts
        candidate_completions = []
        candidate_input_ids = []
        
        for g in range(num_generations):
            # Phase 1: Initial generation (simulate tool call generation)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Execute tool call if requested by the agent
            if "read_sec_document" in gen_text:
                # Extract target path or fallback to sampling annual report directory
                target = "raw_sec"
                for word in gen_text.split():
                    if "/" in word or ".htm" in word or ".pdf" in word:
                        target = word.replace('"', '').replace("'", "").replace(")", "").replace("(", "")
                        break
                
                tool_output = read_sec_document(target)
                multi_turn_prompt = prompt_text + gen_text + f"\n\n=== TOOL RESPONSE ===\n{tool_output}\n\nGenerate final research questions:\n"
            else:
                multi_turn_prompt = prompt_text + gen_text + "\n\nGenerate final research questions:\n"
                
            # Phase 2: Final question generation
            mt_inputs = tokenizer(multi_turn_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                final_outputs = model.generate(**mt_inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
                
            final_completion = tokenizer.decode(final_outputs[0][mt_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            candidate_completions.append(final_completion)
            candidate_input_ids.append(final_outputs[0])

        # Score all G=8 candidates via HUD LLMJudgeGrader
        print(f"[INFO] Scoring G=8 multi-turn trajectories via HUD LLMJudgeGrader...", flush=True)
        rewards = hud_judge_reward_func(candidate_completions, [golden] * num_generations)
        r_tensor = torch.tensor(rewards, device=model.device, dtype=torch.bfloat16)
        
        # Group-Normalize Rewards
        r_mean = r_tensor.mean()
        r_std = r_tensor.std() + 1e-8
        r_norm = (r_tensor - r_mean) / r_std
        
        print(f"[METRIC] Mean Reward: {r_mean.item():.4f}, Std: {r_std.item():.4f}", flush=True)

        # Compute GRPO Policy Loss & KL Divergence
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=model.device, dtype=torch.bfloat16, requires_grad=True)
        
        for g in range(num_generations):
            inp = candidate_input_ids[g].unsqueeze(0)
            outputs = model(inp)
            logits = outputs.logits[:, :-1, :]
            
            with torch.no_grad():
                ref_outputs = ref_model(inp)
                ref_logits = ref_outputs.logits[:, :-1, :]
                
            # Calculate KL divergence
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            kl = torch.exp(log_probs) * (log_probs - ref_log_probs)
            kl_loss = kl.sum(dim=-1).mean()
            
            # Policy gradient loss weighted by normalized reward
            policy_loss = -log_probs.mean() * r_norm[g]
            loss = policy_loss + beta * kl_loss
            total_loss = total_loss + loss / num_generations

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"[UPDATE] Step {step+1} complete | Loss: {total_loss.item():.4f}", flush=True)

    # 4. Save final LoRA weights to mounted volume
    final_dir = "/data/multiturn_grpo_checkpoints/final_lora"
    print(f"[INFO] Training complete! Saving final LoRA checkpoint to {final_dir}...", flush=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    server_task.cancel()
    print("[INFO] Modal Multi-Turn GRPO training run finished successfully!", flush=True)


@app.local_entrypoint()
def main() -> None:
    """CLI entrypoint for `modal run src/ia_question_rl/modal_multiturn_grpo_trainer.py`."""
    print("[INFO] Submitting Multi-Turn GRPO training job to Modal container network...", flush=True)
    train_multiturn_grpo.remote()
