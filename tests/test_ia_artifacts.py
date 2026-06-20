import json

from ia_question_rl.ia_artifacts import context_from_run, discover_artifacts


def test_context_from_run_extracts_questions_metrics_and_gaps(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "layer1_question_pack.json").write_text(
        json.dumps(
            {
                "research_questions": [
                    {"question": "What drove gross margin expansion?"},
                ],
                "metric_families": [{"metric_family": "gross margin"}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "feedback_loop_pack.json").write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "gap_id": "segment_margin_gap",
                        "gap_description": "Segment margin bridge is missing.",
                        "severity": "high",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    artifacts = discover_artifacts(run_dir)
    context = context_from_run(run_dir, ticker="PDD", thesis="Margin durability")

    assert set(artifacts) == {"layer1_question_pack.json", "feedback_loop_pack.json"}
    assert context.existing_questions == ("What drove gross margin expansion?",)
    assert context.metrics == ("gross margin",)
    assert context.evidence_gaps[0].gap_id == "segment_margin_gap"
