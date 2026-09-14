from typing import Literal

from pydantic import BaseModel, Field


class RetrievedEvidence(BaseModel):
    evidence_id: str
    document_id: str
    claim: str
    status: Literal[
        "verified",
        "unverified",
        "disputed",
        "misleading",
    ]
    evidence_type: str
    source_excerpt: str
    score: float
    source: str = "hybrid"


class InvestigatorStep(BaseModel):
    iteration: int
    phase: Literal[
        "investigation",
        "coverage_sweep",
    ] = "investigation"
    query: str
    theory: str
    sufficient: bool
    reason: str
    next_query: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    question: str
    initial_theory: str
    final_theory: str
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    cited_document_ids: list[str] = Field(default_factory=list)
    needs_more_evidence: bool = False
    termination_reason: Literal[
        "sufficient",
        "sufficient_after_coverage",
        "retry_limit",
        "no_next_query",
    ] = "sufficient"
    coverage_sweep_used: bool = False
    coverage_queries: list[str] = Field(default_factory=list)
    trace: list[InvestigatorStep] = Field(default_factory=list)


class SufficiencyDecision(BaseModel):
    sufficient: bool
    reason: str
    revised_theory: str
    next_query: str | None = None


class CoveragePlan(BaseModel):
    gap_summary: str
    queries: list[str] = Field(default_factory=list)


class FinalInvestigatorAnswer(BaseModel):
    final_theory: str
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    cited_document_ids: list[str] = Field(default_factory=list)


class FactCheckResult(BaseModel):
    target_theory: str
    weaknesses: list[str] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    conflicting_timeline_points: list[str] = Field(default_factory=list)
    evidence_for_alternatives: list[str] = Field(default_factory=list)
    challenged_evidence_ids: list[str] = Field(default_factory=list)
    cited_document_ids: list[str] = Field(default_factory=list)
    overall_assessment: str
