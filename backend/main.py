import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from backend.agents import (
    AgentEvidenceStore,
    FactCheckerAgent,
    FaithfulnessEvaluator,
    InvestigatorAgent,
)
from backend.agents.schemas import InvestigationResult
from backend.graph.evidence_graph import EvidenceGraph
from backend.models import CaseEvidenceCorpus


load_dotenv()

app = FastAPI(
    title="CaseFile AI API",
    version="1.1.0",
    description="Interactive agentic RAG investigation backend.",
)

CANONICAL_CASE_PATH = Path("data/processed/canonical_case.json")

_store: AgentEvidenceStore | None = None
_graph: EvidenceGraph | None = None
_case: CaseEvidenceCorpus | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=8, ge=1, le=20)


class InvestigationRequest(BaseModel):
    question: str = Field(min_length=5)
    initial_theory: str | None = None
    max_iterations: int = Field(default=3, ge=1, le=5)
    top_k: int = Field(default=8, ge=3, le=15)


class FactCheckRequest(BaseModel):
    investigation: InvestigationResult
    top_k: int = Field(default=10, ge=3, le=20)


class InterrogationRequest(BaseModel):
    candidate: str = Field(min_length=2)
    question: str = Field(min_length=3)
    top_k: int = Field(default=6, ge=3, le=12)


class GroundedInterrogationAnswer(BaseModel):
    candidate: str
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    caveat: str


class VerdictRequest(BaseModel):
    candidate: str = Field(min_length=2)
    conclusion: str = Field(min_length=10)
    evidence_ids: list[str] = Field(min_length=1)


class VerdictJudgement(BaseModel):
    support_level: Literal[
        "well_supported",
        "partially_supported",
        "weakly_supported",
        "unsupported",
    ]
    assessment: str
    missing_or_conflicting_points: list[str] = Field(default_factory=list)


class VerdictResponse(BaseModel):
    candidate: str
    support_level: str
    citation_validity: float = Field(ge=0.0, le=1.0)
    verified_evidence_count: int
    unverified_evidence_count: int
    other_status_count: int
    unsupported_evidence_ids: list[str] = Field(default_factory=list)
    cited_document_ids: list[str] = Field(default_factory=list)
    assessment: str
    missing_or_conflicting_points: list[str] = Field(default_factory=list)


def get_store() -> AgentEvidenceStore:
    global _store
    if _store is None:
        _store = AgentEvidenceStore()
    return _store


def get_graph() -> EvidenceGraph:
    global _graph
    if _graph is None:
        _graph = EvidenceGraph()
        _graph.build()
    return _graph


def get_case() -> CaseEvidenceCorpus:
    global _case
    if _case is None:
        if not CANONICAL_CASE_PATH.exists():
            raise RuntimeError(
                "Canonical case corpus not found. "
                "Run scripts.build_canonical_evidence first."
            )
        with CANONICAL_CASE_PATH.open("r", encoding="utf-8") as file:
            _case = CaseEvidenceCorpus.model_validate(json.load(file))
    return _case


def get_gemini_client() -> tuple[genai.Client, str]:
    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key not configured.",
        )
    return (
        genai.Client(api_key=api_key),
        os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "CaseFile AI",
        "version": "1.1.0",
    }


@app.get("/case/summary")
def case_summary() -> dict[str, Any]:
    case = get_case()

    people = [
        {
            "entity_id": entity.entity_id,
            "name": entity.canonical_name,
            "aliases": entity.aliases,
        }
        for entity in case.entities
        if entity.entity_type == "person"
    ]

    candidate_names = {
        "John Hector McFarlane",
        "Jonas Oldacre",
    }

    candidates = [
        item
        for item in people
        if item["name"] in candidate_names
    ]

    return {
        "case_id": case.case_id,
        "entities": len(case.entities),
        "evidence": len(case.evidence),
        "relationships": len(case.relationships),
        "people": people,
        "candidates": candidates,
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    results = get_store().search(
        request.query,
        top_k=request.top_k,
    )
    return {
        "query": request.query,
        "results": [
            item.model_dump()
            for item in results
        ],
    }


@app.get("/graph/entity/{entity_id}")
def graph_entity(entity_id: str) -> dict[str, Any]:
    graph = get_graph()
    try:
        return {
            "entity_id": entity_id,
            "evidence": graph.entity_evidence(entity_id),
            "relationships": graph.entity_relationships(entity_id),
            "subgraph": graph.subgraph_for_entity(entity_id),
        }
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc


@app.post("/investigate")
def investigate(
    request: InvestigationRequest,
) -> dict[str, Any]:
    investigator = InvestigatorAgent(
        evidence_store=get_store()
    )
    result = investigator.investigate(
        question=request.question,
        initial_theory=request.initial_theory,
        max_iterations=request.max_iterations,
        top_k=request.top_k,
    )
    return result.model_dump()


@app.post("/fact-check")
def fact_check(
    request: FactCheckRequest,
) -> dict[str, Any]:
    fact_checker = FactCheckerAgent(
        evidence_store=get_store()
    )
    result = fact_checker.fact_check(
        request.investigation,
        top_k=request.top_k,
    )
    return result.model_dump()


@app.post("/faithfulness")
def faithfulness(
    investigation: InvestigationResult,
) -> dict[str, Any]:
    evaluator = FaithfulnessEvaluator(
        evidence_store=get_store()
    )
    return evaluator.evaluate(
        investigation
    ).model_dump()


@app.post("/interrogate")
def interrogate(
    request: InterrogationRequest,
) -> dict[str, Any]:
    store = get_store()

    evidence = store.search(
        f"{request.candidate} {request.question}",
        top_k=request.top_k,
    )

    evidence_text = "\n\n---\n\n".join(
        [
            "\n".join(
                [
                    f"Evidence ID: {item.evidence_id}",
                    f"Document ID: {item.document_id}",
                    f"Status: {item.status}",
                    f"Claim: {item.claim}",
                    f"Excerpt: {item.source_excerpt}",
                ]
            )
            for item in evidence
        ]
    )

    client, model_name = get_gemini_client()

    prompt = f"""
You are answering a detective interrogation question about {request.candidate}.

QUESTION:
{request.question}

RETRIEVED CASE EVIDENCE:
{evidence_text}

Rules:
- Answer only from the supplied evidence.
- Do not roleplay invented memories, motives, or dialogue.
- Preserve uncertainty: verified evidence is stronger than unverified testimony.
- If the evidence cannot answer the question, explicitly say so.
- Cite evidence IDs and document IDs in the answer where useful.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(
                thinking_level="minimal"
            ),
        ),
    )

    result = GroundedInterrogationAnswer(
        candidate=request.candidate,
        answer=response.text or (
            "The available evidence does not support a grounded answer."
        ),
        evidence_ids=[
            item.evidence_id
            for item in evidence
        ],
        document_ids=sorted(
            {
                item.document_id
                for item in evidence
            }
        ),
        caveat=(
            "Grounded evidence answer only. "
            "Unverified claims remain unverified."
        ),
    )

    return result.model_dump()


@app.post("/submit-verdict")
def submit_verdict(
    request: VerdictRequest,
) -> dict[str, Any]:
    store = get_store()

    cited_evidence, missing_ids = store.get_by_ids(
        request.evidence_ids
    )

    if not cited_evidence:
        raise HTTPException(
            status_code=400,
            detail=(
                "None of the submitted evidence IDs exist "
                "in the canonical evidence corpus."
            ),
        )

    valid_count = len(cited_evidence)
    total_count = len(request.evidence_ids)
    citation_validity = valid_count / max(total_count, 1)

    verified = sum(
        item.status == "verified"
        for item in cited_evidence
    )
    unverified = sum(
        item.status == "unverified"
        for item in cited_evidence
    )
    other = valid_count - verified - unverified

    evidence_text = "\n\n---\n\n".join(
        [
            "\n".join(
                [
                    f"Evidence ID: {item.evidence_id}",
                    f"Document ID: {item.document_id}",
                    f"Status: {item.status}",
                    f"Claim: {item.claim}",
                    f"Excerpt: {item.source_excerpt}",
                ]
            )
            for item in cited_evidence
        ]
    )

    client, model_name = get_gemini_client()

    prompt = f"""
Evaluate whether the user's final conclusion is supported by ONLY the cited
case evidence below.

FINAL CANDIDATE / SUBJECT:
{request.candidate}

USER CONCLUSION:
{request.conclusion}

CITED EVIDENCE:
{evidence_text}

Rules:
- Do not use outside Sherlock Holmes knowledge.
- Verified evidence is stronger than unverified testimony.
- Do not call an unverified allegation established fact.
- Judge support quality, not writing quality.
- Mention important conflicts or gaps.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(
                thinking_level="minimal"
            ),
            response_mime_type="application/json",
            response_schema=VerdictJudgement,
        ),
    )

    if response.parsed is not None:
        if isinstance(response.parsed, VerdictJudgement):
            judgement = response.parsed
        else:
            judgement = VerdictJudgement.model_validate(
                response.parsed
            )
    else:
        judgement = VerdictJudgement.model_validate_json(
            response.text
        )

    return VerdictResponse(
        candidate=request.candidate,
        support_level=judgement.support_level,
        citation_validity=citation_validity,
        verified_evidence_count=verified,
        unverified_evidence_count=unverified,
        other_status_count=other,
        unsupported_evidence_ids=missing_ids,
        cited_document_ids=sorted(
            {
                item.document_id
                for item in cited_evidence
            }
        ),
        assessment=judgement.assessment,
        missing_or_conflicting_points=(
            judgement.missing_or_conflicting_points
        ),
    ).model_dump()
