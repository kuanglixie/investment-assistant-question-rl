from ia_question_rl.models import EvidenceGap, ResearchContext
from ia_question_rl.reward import evaluate_question


def test_material_source_grounded_question_beats_vague_question() -> None:
    context = ResearchContext(
        ticker="PDD",
        thesis="Assess Temu margin durability and cash conversion.",
        evidence_gaps=(
            EvidenceGap(
                gap_id="temu_standalone_economics",
                description="Temu standalone unit economics are not disclosed separately.",
            ),
        ),
    )

    strong = evaluate_question(
        "Which official disclosures or segment proxies can test whether Temu standalone unit economics are improving?",
        context,
    )
    weak = evaluate_question("Is this good?", context)

    assert strong.total > weak.total
    assert strong.label in {"useful", "excellent"}
    assert weak.label == "weak"


def test_duplicate_question_loses_novelty() -> None:
    context = ResearchContext(
        ticker="PDD",
        existing_questions=(
            "Which official disclosures can test whether Temu unit economics are improving?",
        ),
    )

    reward = evaluate_question(
        "Which official disclosures can test whether Temu unit economics are improving?",
        context,
    )

    assert reward.components["novelty"] == 0.0
