"""DeepResearch v6 environment: live research tools over an `mcp` capability, LLM-judged.

Tools: search/fetch (live web via Exa) and Sixtyfour enrich_person/enrich_company.
Templates: web_research (a cited web answer) and research_person (a sourced dossier).
"""

# NOTE: do NOT add `from __future__ import annotations` here. Under it, a
# `@env.template` param annotated with Literal/alias/model crashes the
# sync/deploy manifest path (TypeAdapter on a string forward-ref). Keep
# annotations as real objects.
import asyncio
import contextlib
import logging
import os
import socket
import sys
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from hud import Environment
from hud.capabilities import Capability
from hud.graders import LLMJudgeGrader, combine, EvaluationResult

load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[%(levelname)s] %(name)s | %(message)s")
for noisy in ("httpx", "httpcore", "FastMCP", "mcp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("hud-investment")

env = Environment(name="hud-investment")

_MCP_PORT: int | None = None
_MCP_SERVER_TASK: asyncio.Task[None] | None = None


# ── SEC Document & Artifact Reader Tool ───────────────────────────────────────


async def read_sec_document(target_path: str, max_chars: int = 4000) -> str:
    """Read the contents of a local SEC filing document (text, HTML, or PDF) or InvestmentAssistant artifact.

    Accepts absolute paths (e.g., from source_artifacts) or relative paths within the project cache (e.g. data/raw/sec/...).
    """
    def _read_file_content(file_path: str, max_c: int) -> str:
        if file_path.lower().endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                text_parts = []
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
                    if sum(len(p) for p in text_parts) >= max_c:
                        break
                content = "\n".join(text_parts)[:max_c]
                return content or "PDF File is empty"
            except Exception as e:
                return f"Error reading PDF file {file_path}: {e}"
        else:
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    content = f.read(max_c)
                    return content or "File is empty"
            except Exception as e:
                return f"Error reading file {file_path}: {e}"

    if not os.path.isabs(target_path):
        target_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", target_path))
    
    if not os.path.exists(target_path):
        # Fallback check across /data/rl_observations for robust matching during GRPO rollouts
        base_obs = "/data/rl_observations"
        if os.path.exists(base_obs):
            for root, dirs, files in os.walk(base_obs):
                for name in list(dirs) + list(files):
                    if name == os.path.basename(target_path) or os.path.join(root, name).endswith(target_path.lstrip("/")):
                        matched = os.path.join(root, name)
                        if os.path.isdir(matched):
                            return f"Target is a directory. Available files: {os.listdir(matched)}"
                        return _read_file_content(matched, max_chars)

        parent = os.path.dirname(target_path)
        if os.path.exists(parent) and os.path.isdir(parent):
            files = os.listdir(parent)
            return f"File not found: {target_path}. Available files in directory: {files}"
        return f"Path does not exist: {target_path}"
    
    if os.path.isdir(target_path):
        files = os.listdir(target_path)
        return f"Target is a directory. Available files: {files}"
    
    return _read_file_content(target_path, max_chars)


# ── Sixtyfour: deep research on people and companies ──────────────────────────
# Agentic enrichment API (sponsor: sixtyfour.ai). Sync calls take minutes, so the
# tools force the fast `micro` tier and use a long client timeout.

_SIXTYFOUR_BASE = "https://api.sixtyfour.ai"


async def _sixtyfour_post(path: str, payload: dict[str, Any], timeout: float = 900.0) -> dict[str, Any]:
    key = os.getenv("SIXTYFOUR_API_KEY")
    if not key:
        return {"error": "Sixtyfour is not configured. Set SIXTYFOUR_API_KEY to enable deep "
                         "person/company research."}
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{_SIXTYFOUR_BASE}{path}", headers=headers, json=payload)
        if r.status_code >= 500:  # one retry on a transient server error
            r = await client.post(f"{_SIXTYFOUR_BASE}{path}", headers=headers, json=payload)
        logger.info("sixtyfour %s -> %s in %.0fs", path, r.status_code, time.monotonic() - t0)
        if r.status_code >= 400:
            # Return a usable error instead of raising, so the agent can adapt and the
            # rollout doesn't crash on a Sixtyfour-side hiccup.
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text[:300]
            return {"error": f"Sixtyfour returned {r.status_code}", "detail": detail}
        return r.json()


async def enrich_person(name: str, company: str = "", linkedin: str = "") -> dict[str, Any]:
    """Deep-research a person and return a sourced dossier in one call.

    Pass ``company`` and/or ``linkedin`` to disambiguate common names. Returns
    ``structured_data`` (role, company, co-founders, prior companies, sources) plus a
    ``notes`` narrative and ``references``. This is the primary tool for a person dossier.
    """
    lead_info: dict[str, str] = {"name": name}
    if company:
        lead_info["company"] = company
    if linkedin:
        lead_info["linkedin"] = linkedin
    # Keep the struct lean: large structs 500 on Sixtyfour. The response also carries a
    # rich `notes` narrative + `references`, which cover the rest of the dossier.
    struct = {
        "current_role": "Current job title and company",
        "company_description": "What their current company does, in one sentence",
        "cofounders": "Names of their co-founders, if any",
        "prior_companies": "Notable companies or roles before the current one",
        "sources": "List of source URLs the research is based on",
    }
    # tier="micro" keeps the call fast enough for an interactive rollout (low+ can take
    # 5-10 min); the agent can't pick a slower tier.
    return await _sixtyfour_post("/enrich-lead", {"lead_info": lead_info, "struct": struct, "tier": "micro"})


async def enrich_company(company: str, website: str = "") -> dict[str, Any]:
    """Deep-research a company via Sixtyfour. Returns ``structured_data`` + ``confidence_score``."""
    target = f"{company} ({website})" if website else company
    struct = {
        "what_they_do": "One-sentence description of the company",
        "founded_year": "Year the company was founded",
        "headcount": "Approximate number of employees",
        "founders": "Names of the founders",
        "funding": "Funding stage and notable investors, if known",
        "sources": "List of source URLs the research is based on",
    }
    return await _sixtyfour_post(
        "/company-intelligence", {"target_company": target, "struct": struct, "tier": "micro"}
    )


# ── mcp capability lifecycle ──────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _listening(host: str, port: int, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"mcp server never came up on {host}:{port}")


@env.initialize
async def _up() -> None:
    # Import FastMCP lazily so `import tasks` (the task-collection path) stays
    # free of fastmcp/authlib import-time noise.
    from fastmcp import FastMCP

    global _MCP_PORT, _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is None:
        server = FastMCP(name="research-tools")
        server.tool(read_sec_document)
        server.tool(enrich_person)
        server.tool(enrich_company)
        _MCP_PORT = _free_port()
        _MCP_SERVER_TASK = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=_MCP_PORT, show_banner=False)
        )
        await _listening("127.0.0.1", _MCP_PORT)
    env.add_capability(Capability.mcp(name="research", url=f"http://127.0.0.1:{_MCP_PORT}/mcp"))


@env.shutdown
async def _down() -> None:
    global _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is not None:
        _MCP_SERVER_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _MCP_SERVER_TASK
        _MCP_SERVER_TASK = None


# ── tasks ─────────────────────────────────────────────────────────────────────


@env.template()
async def web_research(question: str, answer_should_include: str = "") -> AsyncGenerator[Any, Any]:
    """Answer a question from live web research (Exa); graded by an LLM judge."""
    answer = yield (
        f"{question}\n\nResearch this using web search, then give a direct, specific answer "
        "and cite the source URL you used."
    )

    criterion = (
        f"The answer correctly addresses the question and is consistent with: {answer_should_include}"
        if answer_should_include
        else "The answer correctly and specifically addresses the question, with a cited source."
    )
    result = await combine(
        LLMJudgeGrader.grade(
            weight=1.0, answer=str(answer or ""), criteria=[(criterion, 1.0)], question=question
        )
    )
    logger.info("web_research reward=%.3f", result.reward)
    yield result


@env.template()
async def research_person(
    brief: str, criteria: list[str], ground_truth: str = ""
) -> AsyncGenerator[Any, Any]:
    """Deep-research a person and produce a sourced dossier; graded by an LLM judge.

    The agent uses enrich_person (Sixtyfour) plus search/fetch to build the dossier.

    Args:
        brief: The research brief shown to the agent.
        criteria: Plain-English requirements the dossier must satisfy (one judge
            criterion each, partial credit).
        ground_truth: Verified facts handed to the judge so it can grade accurately.
    """
    answer = yield brief

    crit = [(c, 1.0) for c in criteria]
    question = brief + (
        f"\n\n=== VERIFIED GROUND TRUTH (for grading only) ===\n{ground_truth}" if ground_truth else ""
    )
    result = await combine(
        LLMJudgeGrader.grade(weight=1.0, answer=str(answer or ""), criteria=crit, question=question)
    )
    logger.info("research_person reward=%.3f", result.reward)
    yield result


@env.template()
async def financial_question(task_prompt: str, context_dict: dict[str, Any]) -> AsyncGenerator[Any, Any]:
    """Propose high-quality financial research questions; graded purely by golden question coverage."""
    answer = yield task_prompt

    target_human_questions = tuple(context_dict.get("target_human_questions", []))
    if not target_human_questions:
        yield EvaluationResult(reward=0.0, content="No golden questions provided in context.", info={})
        return

    # Use LLMJudgeGrader to evaluate each golden question individually (each match scores 1)
    golden_questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(target_human_questions))
    criteria = [
        (
            f"The candidate research questions exhibit strong thematic alignment or target the same core business topic (e.g., core business drivers, strategic initiatives, margin trends, or macro industry backdrop) as Golden Question #{i+1}: '{q}'",
            1.0
        )
        for i, q in enumerate(target_human_questions)
    ]
    judge_question = (
        f"Evaluate whether the proposed research questions successfully cover the core business topics and strategic themes present in the golden analyst questions.\n"
        f"Evaluate each golden question independently. Award points (MET) if any candidate question targets the same overarching business topic, financial trend, or strategic initiative (such as international expansion, profit margin fluctuations, or new investment initiatives) as the golden question, even if the candidate uses more rigorous forensic or accounting terminology.\n\n"
        f"=== GOLDEN ANALYST QUESTIONS ===\n{golden_questions_text}"
    )
    judge_result = await combine(
        LLMJudgeGrader.grade(weight=1.0, answer=str(answer or ""), criteria=criteria, question=judge_question)
    )

    # Calculate exact coverage ratio M / N
    total_golden = len(target_human_questions)
    covered_golden = int(round(judge_result.reward * total_golden))
    reward = float(covered_golden) / float(total_golden) if total_golden > 0 else 0.0

    logger.info("financial_question reward=%.3f covered=%d/%d", reward, covered_golden, total_golden)

    content = f"Golden Coverage: {covered_golden}/{total_golden} ({reward*100:.1f}%)\n\n=== AGENT ANSWER ===\n{answer}"

    yield EvaluationResult(reward=reward, content=content, info={"covered_golden": covered_golden, "total_golden": total_golden, "reward": reward})
