from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare train/test GRPO datasets from rl_observations directory.")
    parser.add_argument("--observations-dir", default="/Users/ajing/Downloads/rl_observations", help="Path to rl_observations directory.")
    parser.add_argument("--output-train", default="data/grpo_train_prompts.jsonl", help="Output path for train prompts.")
    parser.add_argument("--output-test", default="data/grpo_test_prompts.jsonl", help="Output path for test prompts.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Ratio of companies to use for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible train/test split.")
    args = parser.parse_args(argv)

    obs_dir = Path(args.observations_dir)
    if not obs_dir.exists():
        print(f"[Error] Observations directory not found: {obs_dir}")
        return 1

    manifest_path = obs_dir / "rl_observation_export_manifest.json"
    companies_data = []

    if manifest_path.exists():
        print(f"[INFO] Loading company manifest from {manifest_path}...")
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
            companies = manifest.get("companies", [])
            for c in companies:
                ticker = c.get("ticker")
                raw_report_count = c.get("raw_report_count", 30)
                raw_sec_company_dir = f"/data/rl_observations/{ticker}/observation/raw_reports"
                companies_data.append((ticker, raw_report_count, raw_sec_company_dir))
    else:
        print(f"[INFO] Manifest not found at {manifest_path}. Scanning subdirectories...")
        for p in obs_dir.iterdir():
            if p.is_dir() and (p / "labels" / "analyst_questions.json").exists():
                companies_data.append((p.name, 30, f"/data/rl_observations/{p.name}/observation/raw_reports"))

    # Sort companies for deterministic splitting
    companies_data.sort(key=lambda x: x[0])
    
    records = []
    for ticker, raw_report_count, raw_sec_company_dir in companies_data:
        labels_file = obs_dir / ticker / "labels" / "analyst_questions.json"
        if not labels_file.exists():
            print(f"[Warning] Labels file not found for {ticker}: {labels_file}")
            continue

        with labels_file.open("r", encoding="utf-8") as f:
            lbl_data = json.load(f)
        
        questions = lbl_data.get("questions", [])
        golden_questions = [q.get("question_text", "").strip() for q in questions if q.get("question_text")]

        if not golden_questions:
            print(f"[Warning] No golden questions found for {ticker}.")
            continue

        system_content = (
            "You are an expert investment research assistant generating high-quality follow-up research questions.\n"
            "Your goal is to propose specific, material, answerable research questions targeting known evidence gaps.\n\n"
            "Rules for high-quality questions:\n"
            "1. Grounding: Explicitly name source types (e.g., '10-K', '20-F', '6-K', 'segment disclosures', 'unit economics', 'filings').\n"
            "2. Answerability: Use testable action verbs ('test', 'verify', 'confirm', 'measure', 'isolate', 'compare').\n"
            f"3. Specificity: Include the company ticker {ticker} and target specific evidence gaps or metric anomalies.\n"
            "4. No Conclusions: Never ask subjective conclusion questions ('should we buy/sell/short/avoid').\n"
            "5. Concise & Singular: Do not chain multiple questions together with excessive conjunctions (and, or, plus).\n"
            "6. Quantity Limit: Generate exactly 3 to 5 high-impact research questions."
        )

        user_content = (
            f"Analyze the attached SEC documents and generate expert research questions.\n\n"
            f"Attachment Summary: Ingested {raw_report_count} local SEC filing exhibits from {raw_sec_company_dir}.\n\n"
            "=== AVAILABLE DOCUMENTS & ARTIFACTS ===\n"
            "You have access to the `read_sec_document` tool to inspect local SEC EDGAR filings and InvestmentAssistant extracted report packs.\n\n"
            "CRITICAL RULES FOR TOOL USAGE:\n"
            "- Do NOT make parallel tool calls. You MUST call `read_sec_document` sequentially (exactly one tool call per turn) to ensure the evaluation harness captures the tool outputs correctly!\n"
            "- You MUST use `read_sec_document` to inspect at least one raw SEC filing file from `annual_reports/` (such as the 10-K/20-F .htm) or `earnings_results/` (such as the 8-K/6-K .htm)!\n"
            "- To avoid looping, do NOT try to read the entire .htm file multiple times. Make exactly ONE call to `read_sec_document` on a raw SEC filing to sample its contents, inspect 1 or 2 extracted report packs if needed, and then immediately generate your final 3-5 research questions!"
        )

        record = {
            "prompt": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "golden_questions": golden_questions,
        }
        records.append((ticker, record))

    print(f"[INFO] Successfully processed {len(records)} companies with golden questions.")

    # Reproducible train/test split
    random.seed(args.seed)
    random.shuffle(records)

    test_size = int(len(records) * args.test_ratio)
    test_records = records[:test_size]
    train_records = records[test_size:]

    train_path = Path(args.output_train)
    test_path = Path(args.output_test)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w", encoding="utf-8") as f:
        for _, rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with test_path.open("w", encoding="utf-8") as f:
        for _, rec in test_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Wrote {len(train_records)} train prompts to {train_path}")
    print(f"[SUCCESS] Wrote {len(test_records)} test prompts to {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
