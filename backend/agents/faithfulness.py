import os
import re
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from backend.agents.evidence_store import AgentEvidenceStore
from backend.agents.schemas import InvestigationResult, RetrievedEvidence


load_dotenv()


class SentenceFaithfulness(BaseModel):
    sentence_index: int
    supported: bool
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str


class FaithfulnessJudgement(BaseModel):
    assessments: list[SentenceFaithfulness] = Field(default_factory=list)
    overall_assessment: str


class CoverageJudgement(BaseModel):
    coverage_status: Literal["pass", "warning", "fail"]
    critical_omission_evidence_ids: list[str] = Field(default_factory=list)
    useful_but_noncritical_evidence_ids: list[str] = Field(default_factory=list)
    coverage_assessment: str


class FaithfulnessResult(BaseModel):
    support_ratio: float = Field(ge=0.0, le=1.0)
    supported_sentences: int
    total_sentences: int
    assessments: list[SentenceFaithfulness]
    overall_assessment: str
    coverage_status: Literal["pass", "warning", "fail"]
    coverage_assessment: str
    critical_omissions: list[RetrievedEvidence] = Field(default_factory=list)
    useful_omissions: list[RetrievedEvidence] = Field(default_factory=list)
    coverage_queries: list[str] = Field(default_factory=list)


def split_sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", text.strip())
        if item.strip()
    ]


class FaithfulnessEvaluator:
    """
    Two-part answer-quality audit.

    Part A — citation grounding:
      checks every sentence only against the evidence IDs the Investigator
      actually cited.

    Part B — coverage:
      independently retrieves additional evidence and asks whether a decisive
      omitted item would materially change the conclusion or resolve an
      explicitly stated uncertainty.

    This distinction prevents a perfectly grounded but incomplete answer from
    receiving a misleading "100% quality" interpretation.
    """

    def __init__(
        self,
        evidence_store: AgentEvidenceStore | None = None,
        model_name: str | None = None,
    ):
        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if not api_key:
            raise ValueError("Gemini API key not found.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or "gemini-3.5-flash-lite"
        )
        self.store = evidence_store or AgentEvidenceStore()

    def _call(self, prompt: str, schema):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"
                ),
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

        if response.parsed is not None:
            if isinstance(response.parsed, schema):
                return response.parsed
            return schema.model_validate(response.parsed)

        return schema.model_validate_json(response.text)

    @staticmethod
    def _format_evidence(evidence) -> str:
        return "\n\n---\n\n".join(
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

    def evaluate(
        self,
        investigation: InvestigationResult,
    ) -> FaithfulnessResult:
        cited_ids = list(
            dict.fromkeys(
                investigation.supporting_evidence_ids
                + investigation.contradicting_evidence_ids
            )
        )

        cited_evidence, _ = self.store.get_by_ids(cited_ids)
        sentences = split_sentences(
            investigation.reasoning_summary
        )

        if not sentences:
            sentence_assessments = []
            support_ratio = 0.0
            supported_count = 0
            grounding_assessment = (
                "No reasoning sentences were available to audit."
            )
        else:
            evidence_text = self._format_evidence(cited_evidence)

            numbered_sentences = "\n".join(
                f"{index}. {sentence}"
                for index, sentence in enumerate(sentences)
            )

            grounding_prompt = f"""
Audit the following Investigator reasoning sentence-by-sentence for factual
faithfulness.

REASONING SENTENCES:
{numbered_sentences}

ALLOWED CITED EVIDENCE:
{evidence_text}

For every sentence index from 0 to {len(sentences) - 1}:
- mark supported=true only if the factual content is supported by the allowed
  evidence above;
- epistemic wording must match the evidence status;
- unverified testimony cannot support wording that treats it as established;
- evidence_ids may contain only IDs visible above;
- cautious synthesis is allowed only when it accurately summarizes the cited
  evidence without inventing facts.

Return one assessment per sentence.
"""

            judgement = self._call(
                grounding_prompt,
                FaithfulnessJudgement,
            )

            valid_ids = {
                item.evidence_id
                for item in cited_evidence
            }

            by_index = {
                assessment.sentence_index: assessment
                for assessment in judgement.assessments
                if 0 <= assessment.sentence_index < len(sentences)
            }

            sentence_assessments: list[SentenceFaithfulness] = []

            for index, _sentence in enumerate(sentences):
                assessment = by_index.get(index)

                if assessment is None:
                    assessment = SentenceFaithfulness(
                        sentence_index=index,
                        supported=False,
                        evidence_ids=[],
                        reason=(
                            "The evaluator did not return an assessment "
                            "for this sentence."
                        ),
                    )
                else:
                    assessment.evidence_ids = [
                        evidence_id
                        for evidence_id in assessment.evidence_ids
                        if evidence_id in valid_ids
                    ]

                sentence_assessments.append(assessment)

            supported_count = sum(
                assessment.supported
                for assessment in sentence_assessments
            )

            support_ratio = (
                supported_count / len(sentence_assessments)
            )
            grounding_assessment = judgement.overall_assessment

        # -------------------------------------------------------------
        # Independent coverage audit.
        # This deliberately searches outside the Investigator's citations.
        # -------------------------------------------------------------
        coverage_queries = list(
            dict.fromkeys(
                [
                    investigation.question,
                    (
                        "direct verified observation resolving the case: "
                        + investigation.question
                    ),
                    (
                        "decisive evidence contradicting this conclusion: "
                        + investigation.final_theory
                    ),
                    (
                        "direct observed event resolving uncertainty in: "
                        + investigation.reasoning_summary
                    ),
                ]
                + investigation.coverage_queries
            )
        )

        coverage_candidates = self.store.search_many(
            coverage_queries,
            top_k=16,
            per_query_k=14,
        )

        # Independent completeness guard. Because the case is small, the audit
        # also inspects every VERIFIED claim that the Investigator did not cite.
        # This prevents a perfectly grounded answer from receiving PASS merely
        # because retrieval failed to surface a decisive direct observation.
        verified_candidates = self.store.verified_evidence()

        cited_set = set(cited_ids)

        omitted_by_candidate_id = {}

        for item in coverage_candidates + verified_candidates:
            if item.evidence_id in cited_set:
                continue
            omitted_by_candidate_id[item.evidence_id] = item

        omitted_candidates = list(
            omitted_by_candidate_id.values()
        )

        omitted_text = self._format_evidence(omitted_candidates)

        coverage_prompt = f"""
Audit the completeness of an Investigator answer.

QUESTION:
{investigation.question}

FINAL THEORY:
{investigation.final_theory}

REASONING:
{investigation.reasoning_summary}

ALREADY CITED EVIDENCE IDS:
{cited_ids}

INDEPENDENTLY RETRIEVED BUT OMITTED CANDIDATE EVIDENCE:
{omitted_text or "None"}

Classify the evidence coverage.

A CRITICAL OMISSION is an omitted evidence item that:
- directly contradicts a material factual statement in the answer; OR
- directly resolves an uncertainty the answer says remains unresolved; OR
- would materially change the verdict or confidence.

Generic completeness rules:
- If the answer says it remains unresolved whether a named person was alive or
  dead, and omitted VERIFIED evidence directly observes that same person
  physically acting or appearing at the relevant time, that evidence is a
  CRITICAL OMISSION.
- If the answer claims there is no direct proof of a fact but omitted VERIFIED
  evidence directly observes that fact, that evidence is a CRITICAL OMISSION.
- Verified direct observations outrank theories, testimony, absence-of-evidence
  arguments, and circumstantial inferences for resolving the specific fact
  they directly observe.

Do NOT call an item critical merely because it is relevant or adds detail.

coverage_status:
- pass: no material omission found;
- warning: useful omitted evidence exists but does not change the conclusion;
- fail: at least one critical omission exists.

Return only evidence IDs visible in the omitted candidate evidence.
"""

        coverage_judgement = self._call(
            coverage_prompt,
            CoverageJudgement,
        )

        omitted_by_id = {
            item.evidence_id: item
            for item in omitted_candidates
        }

        critical_ids = [
            item
            for item in coverage_judgement.critical_omission_evidence_ids
            if item in omitted_by_id
        ]
        useful_ids = [
            item
            for item in coverage_judgement.useful_but_noncritical_evidence_ids
            if item in omitted_by_id and item not in critical_ids
        ]

        coverage_status = coverage_judgement.coverage_status
        if critical_ids:
            coverage_status = "fail"
        elif coverage_status == "fail":
            # Do not report FAIL without a validated critical evidence ID.
            coverage_status = "warning"

        return FaithfulnessResult(
            support_ratio=support_ratio,
            supported_sentences=supported_count,
            total_sentences=len(sentences),
            assessments=sentence_assessments,
            overall_assessment=grounding_assessment,
            coverage_status=coverage_status,
            coverage_assessment=coverage_judgement.coverage_assessment,
            critical_omissions=[
                omitted_by_id[item]
                for item in critical_ids
            ],
            useful_omissions=[
                omitted_by_id[item]
                for item in useful_ids
            ],
            coverage_queries=coverage_queries,
        )
