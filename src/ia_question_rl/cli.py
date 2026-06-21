from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from ia_question_rl.baseline_policy import propose_questions
from ia_question_rl.ia_artifacts import context_from_run
from ia_question_rl.models import EvidenceGap, ResearchContext
from ia_question_rl.reward import evaluate_question
from ia_question_rl.transcript_questions import (
    extract_analyst_questions_from_reader_url,
    extract_analyst_questions_from_url,
    write_questions_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ia-question-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score one candidate research question.")
    score_parser.add_argument("--question", required=True)
    score_parser.add_argument("--ticker", required=True)
    score_parser.add_argument("--company-name")
    score_parser.add_argument("--thesis")
    score_parser.add_argument("--gap", action="append", default=[])
    score_parser.add_argument("--existing-question", action="append", default=[])
    score_parser.add_argument("--target-human-question", action="append", default=[])

    extract_parser = subparsers.add_parser(
        "extract-episode",
        help="Read an InvestmentAssistant run directory and write one question-RL episode.",
    )
    extract_parser.add_argument("--run-dir", required=True)
    extract_parser.add_argument("--ticker", required=True)
    extract_parser.add_argument("--company-name")
    extract_parser.add_argument("--thesis")
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--max-questions", type=int, default=5)

    transcript_parser = subparsers.add_parser(
        "extract-analyst-questions",
        help="Extract analyst questions from an earnings-call transcript URL.",
    )
    transcript_parser.add_argument("--url", action="append", required=True)
    transcript_parser.add_argument("--ticker")
    transcript_parser.add_argument("--output", required=True)
    transcript_parser.add_argument(
        "--reader",
        action="store_true",
        help="Fetch transcript text through the Jina Reader endpoint before extracting questions.",
    )

    args = parser.parse_args(argv)

    if args.command == "score":
        return _score(args)
    if args.command == "extract-episode":
        return _extract_episode(args)
    if args.command == "extract-analyst-questions":
        return _extract_analyst_questions(args)
    raise ValueError(f"Unsupported command: {args.command}")


def _score(args: argparse.Namespace) -> int:
    context = ResearchContext(
        ticker=args.ticker,
        company_name=args.company_name,
        thesis=args.thesis,
        evidence_gaps=tuple(
            EvidenceGap(gap_id=f"manual_gap_{index + 1}", description=gap)
            for index, gap in enumerate(args.gap)
        ),
        existing_questions=tuple(args.existing_question),
        target_human_questions=tuple(getattr(args, "target_human_question", [])),
    )
    reward = evaluate_question(args.question, context)
    print(json.dumps(reward.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _extract_episode(args: argparse.Namespace) -> int:
    context = context_from_run(
        args.run_dir,
        ticker=args.ticker,
        thesis=args.thesis,
        company_name=args.company_name,
    )
    candidates = propose_questions(context, max_questions=args.max_questions)
    rewarded_candidates = []
    for candidate in candidates:
        payload = candidate.to_dict()
        payload["reward"] = evaluate_question(candidate.question, context).to_dict()
        rewarded_candidates.append(payload)

    episode = {
        "episode_id": str(uuid.uuid4()),
        "source_run_dir": str(Path(args.run_dir).resolve()),
        "context": context.to_dict(),
        "candidates": rewarded_candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    print(json.dumps({"wrote": str(output), "candidate_count": len(rewarded_candidates)}, indent=2))
    return 0


def _extract_analyst_questions(args: argparse.Namespace) -> int:
    questions = []
    for url in args.url:
        extractor = extract_analyst_questions_from_reader_url if args.reader else extract_analyst_questions_from_url
        questions.extend(extractor(url, ticker=args.ticker))
    write_questions_jsonl(questions, args.output)
    print(json.dumps({"wrote": args.output, "question_count": len(questions)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
