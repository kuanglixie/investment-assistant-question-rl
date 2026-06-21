from __future__ import annotations

import html
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


USER_AGENT = "investment-assistant-question-rl/0.1 (+research prototype)"
JINA_READER_PREFIX = "https://r.jina.ai/http://"


@dataclass(frozen=True)
class AnalystQuestion:
    ticker: str | None
    source_url: str
    analyst: str
    firm: str | None
    question_text: str
    sequence_index: int
    source: str = "motley_fool"
    schema_version: str = "analyst_question.v0"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    text: str
    previous_operator_text: str | None = None
    role: str | None = None


def fetch_transcript_html(url: str, timeout: int = 30) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def fetch_transcript_markdown(url: str, timeout: int = 60) -> str:
    reader_url = f"{JINA_READER_PREFIX}{url}"
    request = urllib.request.Request(reader_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def extract_analyst_questions_from_url(url: str, ticker: str | None = None) -> list[AnalystQuestion]:
    return extract_analyst_questions_from_html(fetch_transcript_html(url), source_url=url, ticker=ticker)


def extract_analyst_questions_from_reader_url(url: str, ticker: str | None = None) -> list[AnalystQuestion]:
    return extract_analyst_questions_from_markdown(fetch_transcript_markdown(url), source_url=url, ticker=ticker)


def extract_analyst_questions_from_html(
    transcript_html: str,
    source_url: str,
    ticker: str | None = None,
) -> list[AnalystQuestion]:
    company_participants = _extract_company_participants(transcript_html)
    turns = _extract_transcript_turns(transcript_html)
    return _questions_from_turns(turns, company_participants, source_url=source_url, ticker=ticker)


def extract_analyst_questions_from_markdown(
    transcript_markdown: str,
    source_url: str,
    ticker: str | None = None,
) -> list[AnalystQuestion]:
    company_participants = _extract_company_participants_from_markdown(transcript_markdown)
    turns = _extract_transcript_turns_from_markdown(transcript_markdown)
    return _questions_from_turns(turns, company_participants, source_url=source_url, ticker=ticker)


def _questions_from_turns(
    turns: list[SpeakerTurn],
    company_participants: set[str],
    *,
    source_url: str,
    ticker: str | None,
) -> list[AnalystQuestion]:
    qna_turns = _turns_after_qna_start(turns)

    questions: list[AnalystQuestion] = []
    for turn in qna_turns:
        embedded_questions = _embedded_analyst_questions(turn)
        for analyst, firm, question_text in embedded_questions:
            questions.append(
                AnalystQuestion(
                    ticker=ticker,
                    source_url=source_url,
                    analyst=analyst,
                    firm=firm,
                    question_text=question_text,
                    sequence_index=len(questions) + 1,
                )
            )
        if embedded_questions:
            continue
        if not _is_analyst_question_turn(turn, company_participants):
            continue
        firm = _firm_from_operator_intro(turn.previous_operator_text or "", turn.speaker)
        questions.append(
            AnalystQuestion(
                ticker=ticker,
                source_url=source_url,
                analyst=_clean_analyst_name(turn.speaker),
                firm=firm,
                question_text=turn.text,
                sequence_index=len(questions) + 1,
            )
        )
    return questions


def write_questions_jsonl(questions: list[AnalystQuestion], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")


def _extract_company_participants(page_html: str) -> set[str]:
    section = _between_heading(page_html, "call-participants")
    participants: set[str] = {"operator"}
    for item_html in re.findall(r"<li[^>]*>(.*?)</li>", section, flags=re.I | re.S):
        text = _clean_text(item_html)
        if not text:
            continue
        name = text.split("—")[-1].split(" - ")[-1].strip()
        if name:
            participants.add(name.lower())
    return participants


def _extract_company_participants_from_markdown(markdown_text: str) -> set[str]:
    section = _markdown_section(markdown_text, "CALL PARTICIPANTS")
    participants: set[str] = {"operator"}
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("*"):
            continue
        text = _clean_markdown_text(stripped.lstrip("*").strip())
        name = re.split(r"\s+[—–-]\s+", text)[-1].strip()
        if name:
            participants.add(name.lower())
    return participants


def _extract_transcript_turns(page_html: str) -> list[SpeakerTurn]:
    transcript = _between_heading(page_html, "full-conference-call-transcript") or page_html
    turns: list[SpeakerTurn] = []
    previous_operator_text: str | None = None

    pattern = re.compile(r"<p[^>]*>\s*<strong>(.*?):</strong>\s*(.*?)</p>", flags=re.I | re.S)
    for match in pattern.finditer(transcript):
        speaker = _clean_text(match.group(1))
        text = _clean_text(match.group(2))
        if not speaker or not text:
            continue
        turn = SpeakerTurn(speaker=speaker, text=text, previous_operator_text=previous_operator_text)
        turns.append(turn)
        if speaker.lower() == "operator":
            previous_operator_text = text
    return turns


def _extract_transcript_turns_from_markdown(markdown_text: str) -> list[SpeakerTurn]:
    transcript = (
        _markdown_section(markdown_text, "Full Conference Call Transcript")
        or _markdown_section(markdown_text, "Questions & Answers")
        or markdown_text
    )
    turns: list[SpeakerTurn] = []
    previous_operator_text: str | None = None

    pattern = re.compile(
        r"^\*\*(?P<colon_speaker>[^*\n:]{1,100}):\*\*\s*(?P<inline_text>.*)$"
        r"|^\*\*(?P<heading_speaker>[^*\n:]{1,100})\*\*(?:\s+--\s+_(?P<role>[^_\n]+)_)?\s*$",
        flags=re.M,
    )
    matches = list(pattern.finditer(transcript))
    for index, match in enumerate(matches):
        speaker = _clean_markdown_text(match.group("colon_speaker") or match.group("heading_speaker") or "")
        role = _clean_markdown_text(match.group("role") or "") or None
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(transcript)
        inline_text = match.group("inline_text") or ""
        text = _clean_markdown_text(f"{inline_text}\n{transcript[match.end():next_start]}")
        if not speaker or not text:
            continue
        turn = SpeakerTurn(speaker=speaker, text=text, previous_operator_text=previous_operator_text, role=role)
        turns.append(turn)
        if speaker.lower() == "operator":
            previous_operator_text = text
    return turns


def _turns_after_qna_start(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    for index, turn in enumerate(turns):
        combined = f"{turn.speaker} {turn.text}".lower()
        if "open the call for questions" in combined:
            return turns[index + 1 :]
        if "open the call to questions" in combined:
            return turns[index + 1 :]
        if "first question comes from" in combined:
            return turns[index:]
        if "question comes from" in combined and turn.speaker.lower() == "operator":
            return turns[index:]
    return turns


def _is_analyst_question_turn(turn: SpeakerTurn, company_participants: set[str]) -> bool:
    speaker = _clean_analyst_name(turn.speaker).lower()
    if speaker in company_participants or speaker in {"operator", "unknown speaker"}:
        return False
    if turn.role and "analyst" not in turn.role.lower():
        return False
    word_count = len(turn.text.split())
    if word_count < 12:
        return False
    if not _looks_question_like(turn.text) and word_count < 25:
        return False
    return True


def _looks_question_like(text: str) -> bool:
    lowered = text.lower()
    cues = (
        "?",
        "can you",
        "could you",
        "how do",
        "how should",
        "what are",
        "what is",
        "what do",
        "why",
        "where",
        "would you",
        "maybe",
        "talk about",
        "elaborate",
        "help us",
        "give us",
    )
    return any(cue in lowered for cue in cues)


def _firm_from_operator_intro(operator_text: str, analyst_name: str) -> str | None:
    text = _clean_text(operator_text)
    if not text:
        return None
    analyst = re.escape(_clean_analyst_name(analyst_name))
    boundary = r"(?=\.?\s*(?:please|(?:[A-Z][a-z]+,\s+)?your line|you may|go ahead|followed by|$))"
    patterns = (
        rf"{analyst}\s+with\s+(.+?){boundary}",
        rf"{analyst}\s+from\s+(.+?){boundary}",
        rf"take (?:our )?(?:first|next) from {analyst}\s+with\s+(.+?){boundary}",
        rf"go (?:ahead and )?(?:take|to) (?:our )?(?:first|next)?\s*(?:question from )?{analyst}\s+with\s+(.+?){boundary}",
        rf"line of [^.;]+ with (.+?){boundary}",
        rf"question comes from [^.;]+ with (.+?){boundary}",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return _clean_firm_name(match.group(1))
    return None


def _clean_analyst_name(speaker: str) -> str:
    speaker = re.sub(r"^analyst\s*\((.*?)\)$", r"\1", speaker.strip(), flags=re.I)
    return _clean_text(speaker)


def _embedded_analyst_questions(turn: SpeakerTurn) -> list[tuple[str, str | None, str]]:
    text = turn.text
    patterns = (
        r"from\s+(?P<analyst>[A-Z][A-Za-z .'-]+?)\s+(?:of|from|with)\s+(?P<firm>.+?)\.\s+"
        r"(?:his|her|their)\s+question\s+is:\s*(?P<question>.+)",
        r"(?:question\s+(?:comes|is)\s+from|question\s+from|from)\s+"
        r"(?P<analyst>[A-Z][A-Za-z .'-]+?)\s+(?:of|from|with)\s+"
        r"(?P<firm>[A-Z][A-Za-z0-9&.,' -]{1,80}?):\s*(?P<question>.+)",
    )

    embedded: list[tuple[str, str | None, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            firm = _clean_firm_name(match.group("firm"))
            if "question is" in firm.lower():
                continue
            question = _clean_text(match.group("question"))
            if not _looks_question_like(question):
                continue
            embedded.append(
                (
                    _clean_analyst_name(match.group("analyst")),
                    firm or None,
                    question,
                )
            )
        if embedded:
            return embedded
    return embedded


def _markdown_section(markdown_text: str, heading: str) -> str:
    start_match = re.search(rf"^##\s+{re.escape(heading)}:?\s*$", markdown_text, flags=re.I | re.M)
    if not start_match:
        return ""
    next_heading = re.search(r"^##\s+", markdown_text[start_match.end() :], flags=re.M)
    end = start_match.end() + next_heading.start() if next_heading else len(markdown_text)
    return markdown_text[start_match.end() : end]


def _between_heading(page_html: str, heading_id: str) -> str:
    start_match = re.search(rf"<h2[^>]*id=[\"']{re.escape(heading_id)}[\"'][^>]*>", page_html, flags=re.I)
    if not start_match:
        return ""
    next_heading = re.search(r"<h2[^>]*id=[\"'][^\"']+[\"'][^>]*>", page_html[start_match.end() :], flags=re.I)
    end = start_match.end() + next_heading.start() if next_heading else len(page_html)
    return page_html[start_match.end() : end]


def _clean_text(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", " ", fragment, flags=re.I)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html.unescape(fragment)).strip()


def _clean_markdown_text(fragment: str) -> str:
    fragment = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", fragment)
    fragment = re.sub(r"[*_`]+", "", fragment)
    return re.sub(r"\s+", " ", html.unescape(fragment)).strip()


def _clean_firm_name(firm: str) -> str:
    firm = _clean_text(firm)
    firm = re.split(r"\b(?:please proceed|your line is open|go ahead)\b", firm, flags=re.I)[0]
    return firm.strip(" .")
