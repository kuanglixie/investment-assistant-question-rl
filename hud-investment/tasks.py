"""Shipped taskset for the hud-investment environment.

Evaluates financial question generation models against golden analyst benchmarks.
"""

import json
import os
from env import env, financial_question  # noqa: F401  (re-export env for `hud eval tasks.py`)

# Financial Question Task (InvestmentAssistant Question RL)
_episodes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/episodes"))

with open(os.path.join(_episodes_dir, "pdd_supervised_grpo_task.json"), encoding="utf-8") as f:
    _grpo_task = json.load(f)

with open(os.path.join(_episodes_dir, "pdd_refactored_targets.jsonl"), encoding="utf-8") as f:
    _pdd_episode = json.loads(f.readline())

# Override target_human_questions with the true gold_questions for proper evaluation alignment
_pdd_episode["context"]["target_human_questions"] = _grpo_task["gold_questions"]

# Inject available documents and source artifacts into the prompt so the agent knows what to read
_attachment_dir = _grpo_task.get("attachment", "data/raw/sec/0001737806/documents")
_source_artifacts = "\n".join(f"- {p}" for p in _pdd_episode["context"].get("source_artifacts", []))

# Automatically discover and categorize primary SEC financial document files
_base_sec_dir = os.path.abspath(os.path.join(_episodes_dir, "..", _attachment_dir))
_annual_reports = []
_quarterly_reports = []
_earnings_results = []
_proxy_governance = []

if os.path.exists(_base_sec_dir):
    for root, dirs, files in os.walk(_base_sec_dir):
        for file in sorted(files):
            if not file.startswith(".") and not file.endswith(".metadata.json"):
                rel_path = os.path.relpath(os.path.join(root, file), os.path.abspath(os.path.join(_episodes_dir, "..")))
                parts = rel_path.split(os.sep)
                # Form type is typically the subdirectory under documents (e.g. 20-F, 6-K, 10-K, 10-Q, DEF 14A)
                form_type = parts[5] if len(parts) > 5 else (parts[4] if len(parts) > 4 else "")
                
                if form_type in ("10-K", "10-K_A", "20-F", "20-F_A"):
                    _annual_reports.append(f"  - {rel_path}")
                elif form_type in ("10-Q", "10-Q_A"):
                    _quarterly_reports.append(f"  - {rel_path}")
                elif form_type in ("8-K", "8-K_A", "6-K", "6-K_A"):
                    _earnings_results.append(f"  - {rel_path}")
                elif form_type in ("DEF 14A", "PRE 14A", "DEF 14C", "PROXY"):
                    _proxy_governance.append(f"  - {rel_path}")

_sec_categories_str = (
    f"annual_reports/ (10-K, 10-K/A, 20-F, 20-F/A):\n" + ("\n".join(_annual_reports) if _annual_reports else "  (None available)") + "\n\n"
    f"quarterly_reports/ (10-Q, 10-Q/A):\n" + ("\n".join(_quarterly_reports) if _quarterly_reports else "  (None available)") + "\n\n"
    f"earnings_results/ (8-K/6-K earnings & financial results):\n" + ("\n".join(_earnings_results) if _earnings_results else "  (None available)") + "\n\n"
    f"proxy_governance/:\n" + ("\n".join(_proxy_governance) if _proxy_governance else "  (None available)")
)

_enhanced_prompt_with_sec = (
    f"{_grpo_task['task_prompt']}\n\n"
    f"=== AVAILABLE DOCUMENTS & ARTIFACTS ===\n"
    f"You have access to the `read_sec_document` tool to inspect the following local files and directories:\n"
    f"1. SEC EDGAR Filings Directory: {_attachment_dir}\n"
    f"Available SEC Financial Documents:\n{_sec_categories_str}\n\n"
    f"2. InvestmentAssistant Extracted Report Packs:\n{_source_artifacts}\n\n"
    f"CRITICAL RULES FOR TOOL USAGE:\n"
    f"- Do NOT make parallel tool calls. You MUST call `read_sec_document` sequentially (exactly one tool call per turn) to ensure the evaluation harness captures the tool outputs correctly!\n"
    f"- You MUST use `read_sec_document` to inspect at least one raw SEC filing file from `annual_reports/` (such as the 20-F .htm) or `earnings_results/` (such as the 6-K .htm)!\n"
    f"- To avoid looping, do NOT try to read the entire .htm file multiple times. Make exactly ONE call to `read_sec_document` on a raw SEC filing to sample its contents, inspect 1 or 2 extracted report packs if needed, and then immediately generate your final 3-5 research questions!"
)

_enhanced_prompt_no_sec = (
    f"{_grpo_task['task_prompt']}\n\n"
    f"=== AVAILABLE ARTIFACTS ===\n"
    f"You have access to the `read_sec_document` tool to inspect the following local files:\n"
    f"InvestmentAssistant Extracted Report Packs:\n{_source_artifacts}\n\n"
    f"CRITICAL RULES FOR TOOL USAGE:\n"
    f"- Do NOT make parallel tool calls. You MUST call `read_sec_document` sequentially (exactly one tool call per turn) to ensure the evaluation harness captures the tool outputs correctly!\n"
    f"- Inspect the extracted report packs (financial_report_pack.json, source_map.json, etc.) to gather your evidence.\n"
    f"- Inspect 2 or 3 key files at most, then immediately generate your final 3-5 research questions!"
)

_enhanced_prompt_no_artifacts = _grpo_task["task_prompt"]

_pdd_financial_question_with_sec = financial_question(
    task_prompt=_enhanced_prompt_with_sec,
    context_dict=_pdd_episode["context"],
)
_pdd_financial_question_with_sec.slug = "pdd-financial-question-with-sec"

_pdd_financial_question_no_sec = financial_question(
    task_prompt=_enhanced_prompt_no_sec,
    context_dict=_pdd_episode["context"],
)
_pdd_financial_question_no_sec.slug = "pdd-financial-question-no-sec"

_pdd_financial_question_no_artifacts = financial_question(
    task_prompt=_enhanced_prompt_no_artifacts,
    context_dict=_pdd_episode["context"],
)
_pdd_financial_question_no_artifacts.slug = "pdd-financial-question-no-artifacts"

_enhanced_prompt_aligned_mock = (
    "You are an aligned expert investment research assistant generating high-quality follow-up research questions for PDD Holdings (PDD).\n"
    "Generate exactly 5 specific, material, answerable research questions targeting the following core strategic business themes:\n"
    "1. The strategic considerations behind the 3-year RMB 100 billion investment plan for the first-party brand initiative, specifically where investments will be allocated and when they will reflect in financials.\n"
    "2. PDD's global business user growth, focusing on whether user growth has met expectations and how the company plans to retain and serve global consumers.\n"
    "3. The macro consumer backdrop from National Bureau of Statistics data, specifically asking where future growth for GMV and online marketing services will come from given solid physical goods online penetration.\n"
    "4. PDD's strategic view and layouts regarding emerging e-commerce business models, specifically live e-commerce and instant retail.\n"
    "5. The long-term perspective on first-quarter fluctuations in the cost-to-profit ratio and how to predict PDD's stable profit margin levels."
)

_pdd_financial_question_aligned_mock = financial_question(
    task_prompt=_enhanced_prompt_aligned_mock,
    context_dict=_pdd_episode["context"],
)
_pdd_financial_question_aligned_mock.slug = "pdd-financial-question-aligned-mock"

tasks = [_pdd_financial_question_with_sec, _pdd_financial_question_no_sec, _pdd_financial_question_no_artifacts, _pdd_financial_question_aligned_mock]
