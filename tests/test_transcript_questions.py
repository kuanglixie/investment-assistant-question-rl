from ia_question_rl.transcript_questions import (
    extract_analyst_questions_from_html,
    extract_analyst_questions_from_markdown,
)


SAMPLE_MOTLEY_FOOL_HTML = """
<h2 id="call-participants">Call participants</h2>
<ul>
  <li>Chief Executive Officer — Jane CEO</li>
  <li>Chief Financial Officer — Pat CFO</li>
  <li>Head of Investor Relations — Riley IR</li>
</ul>
<h2 id="full-conference-call-transcript">Full Conference Call Transcript</h2>
<p><strong>Riley IR:</strong> We will now open the call for questions.</p>
<p><strong>Operator:</strong> Our first question comes from the line of Alex Smith with Morgan Stanley. Please proceed.</p>
<p><strong>Alex Smith:</strong> Can you elaborate on whether AI demand is changing renewal cycles and commercial bookings?</p>
<p><strong>Pat CFO:</strong> Thanks, Alex. Let me start with bookings.</p>
<p><strong>Operator:</strong> The next question comes from the line of Blake Chen with UBS. Please proceed.</p>
<p><strong>Blake Chen:</strong> How should we think about the durability of margin expansion as capex steps higher?</p>
<p><strong>Jane CEO:</strong> We are focused on long-term returns.</p>
<p><strong>Blake Chen:</strong> Thank you.</p>
"""


def test_extract_analyst_questions_from_motley_fool_style_html() -> None:
    questions = extract_analyst_questions_from_html(SAMPLE_MOTLEY_FOOL_HTML, "https://example.test", ticker="MSFT")

    assert [question.analyst for question in questions] == ["Alex Smith", "Blake Chen"]
    assert questions[0].firm == "Morgan Stanley"
    assert questions[1].firm == "UBS"
    assert questions[0].ticker == "MSFT"
    assert "commercial bookings" in questions[0].question_text


def test_extract_analyst_questions_from_reader_markdown() -> None:
    markdown = """
## CALL PARTICIPANTS

*   Chief Executive Officer — Jane CEO
*   Chief Financial Officer — Pat CFO
*   Head of Investor Relations — Riley IR

## Full Conference Call Transcript

**Riley IR:** We will now open the call for questions.

**Operator:** Certainly. We will go ahead and take our first from Alex Smith with Morgan Stanley.

**Alex Smith:** Can you elaborate on whether AI demand is changing renewal cycles and commercial bookings?

**Pat CFO:** Thanks, Alex. Let me start with bookings.

**Operator:** We will take our next question from Blake Chen with UBS. Please proceed.

**Blake Chen:** How should we think about the durability of margin expansion as capex steps higher?

**Jane CEO:** We are focused on long-term returns.
"""

    questions = extract_analyst_questions_from_markdown(markdown, "https://example.test", ticker="AAPL")

    assert [question.analyst for question in questions] == ["Alex Smith", "Blake Chen"]
    assert questions[0].firm == "Morgan Stanley"
    assert questions[1].firm == "UBS"
    assert questions[0].ticker == "AAPL"


def test_extract_analyst_questions_from_legacy_reader_markdown() -> None:
    markdown = """
## Questions & Answers:

**Operator**

Our first question comes from Andrew Schmidt from Citi. Please go ahead. Your line is open.

**Andrew Schmidt** -- _Analyst_

Hi. Can you talk about cross-border growth drivers and your expectations for next year?

**Pat CFO** -- _Chief Financial Officer_

Sure. Let me start with cross-border trends.

**Operator**

Our next question comes from Lisa Gill with J.P. Morgan. Please proceed with your question.

**Lisa Gill** -- _Analyst_

Thanks. Could you discuss margin recovery and how we should think about the guidance range?

**Jane CEO** -- _Chief Executive Officer_

Thanks for the question.
"""

    questions = extract_analyst_questions_from_markdown(markdown, "https://example.test", ticker="MA")

    assert [question.analyst for question in questions] == ["Andrew Schmidt", "Lisa Gill"]
    assert questions[0].firm == "Citi"
    assert questions[1].firm == "J.P. Morgan"


def test_extract_embedded_moderator_read_analyst_questions() -> None:
    markdown = """
## Full Conference Call Transcript

**Investor Relations:** We will now take questions from the queue.

**Spencer Wong:** Following up on that question, we have one from Sean Diffely of Morgan Stanley. His question is: What have been your biggest learnings from the transaction experience, and does it change your appetite for M&A?

**Chief Executive Officer:** We learned a lot from the process.

**Spencer Wong:** Our next question comes from Vikram Kesavabhotla of Baird: How is your engagement quality metric performing so far this year?

**Chief Executive Officer:** The metric continues to improve.
"""

    questions = extract_analyst_questions_from_markdown(markdown, "https://example.test", ticker="NFLX")

    assert [question.analyst for question in questions] == ["Sean Diffely", "Vikram Kesavabhotla"]
    assert questions[0].firm == "Morgan Stanley"
    assert questions[1].firm == "Baird"
