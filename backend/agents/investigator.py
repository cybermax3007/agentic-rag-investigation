import json
import os
from collections import OrderedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.agents.evidence_store import AgentEvidenceStore
from backend.agents.schemas import (
    FinalInvestigatorAnswer,
    InvestigationResult,
    InvestigatorStep,
    SufficiencyDecision,
)


load_dotenv()


INVESTIGATOR_SYSTEM_PROMPT = """
You are the Investigator Agent in CaseFile AI.

You must reason only from the supplied retrieved evidence.

Rules:
- Treat verified evidence as stronger than unverified testimony.
- Never convert unverified testimony into established fact.
- Explicitly consider evidence that conflicts with the current theory.
- If evidence is insufficient, say so and request a better retrieval query.
- Do not invent facts, evidence IDs, document IDs, people, events, or motives.
- Every factual conclusion in the final answer must be grounded in supplied
  evidence IDs and document IDs.
- A theory may be plausible without being proven.
"""


class InvestigatorAgent:
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
            raise ValueError(
                "Gemini API key not found."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or "gemini-3.5-flash-lite"
        )
        self.store = evidence_store or AgentEvidenceStore()

    def _structured_call(
        self,
        prompt: str,
        schema,
    ):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=INVESTIGATOR_SYSTEM_PROMPT,
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
            return schema.model_validate(
                response.parsed
            )

        return schema.model_validate_json(
            response.text
        )

    @staticmethod
    def _format_evidence(
        evidence,
    ) -> str:
        blocks = []

        for item in evidence:
            blocks.append(
                "\n".join(
                    [
                        f"Evidence ID: {item.evidence_id}",
                        f"Document ID: {item.document_id}",
                        f"Status: {item.status}",
                        f"Type: {item.evidence_type}",
                        f"Claim: {item.claim}",
                        f"Excerpt: {item.source_excerpt}",
                    ]
                )
            )

        return "\n\n---\n\n".join(blocks)

    def investigate(
        self,
        question: str,
        initial_theory: str | None = None,
        max_iterations: int = 3,
        top_k: int = 8,
    ) -> InvestigationResult:
        theory = (
            initial_theory
            or f"Initial working theory for: {question}"
        )

        query = question
        trace: list[InvestigatorStep] = []

        seen_evidence = OrderedDict()

        for iteration in range(
            1,
            max_iterations + 1,
        ):
            retrieved = self.store.search(
                query=query,
                top_k=top_k,
            )

            for item in retrieved:
                seen_evidence[
                    item.evidence_id
                ] = item

            evidence_text = self._format_evidence(
                list(seen_evidence.values())
            )

            decision_prompt = f"""
QUESTION:
{question}

CURRENT THEORY:
{theory}

RETRIEVED EVIDENCE:
{evidence_text}

Evaluate whether the current evidence is sufficient to answer the question
responsibly.

If insufficient:
- explain the precise gap;
- revise the working theory if needed;
- produce one focused next_query designed to retrieve missing or adversarial
  evidence.

If sufficient:
- next_query must be null.
"""

            decision = self._structured_call(
                decision_prompt,
                SufficiencyDecision,
            )

            trace.append(
                InvestigatorStep(
                    iteration=iteration,
                    query=query,
                    theory=theory,
                    sufficient=decision.sufficient,
                    reason=decision.reason,
                    next_query=decision.next_query,
                    evidence_ids=[
                        item.evidence_id
                        for item in retrieved
                    ],
                    document_ids=sorted(
                        {
                            item.document_id
                            for item in retrieved
                        }
                    ),
                )
            )

            theory = decision.revised_theory

            if decision.sufficient:
                break

            if not decision.next_query:
                break

            query = decision.next_query

        all_evidence = list(
            seen_evidence.values()
        )

        evidence_text = self._format_evidence(
            all_evidence
        )

        final_prompt = f"""
QUESTION:
{question}

INITIAL THEORY:
{initial_theory or "None supplied"}

FINAL WORKING THEORY:
{theory}

AGENT TRACE:
{json.dumps([step.model_dump() for step in trace], indent=2)}

AVAILABLE EVIDENCE:
{evidence_text}

Return the final grounded investigation result.

Requirements:
- verdict must be concise and cautious;
- reasoning_summary must distinguish verified facts from unverified claims;
- supporting_evidence_ids and contradicting_evidence_ids may contain ONLY
  evidence IDs visible above;
- cited_document_ids may contain ONLY document IDs visible above;
- confidence is epistemic confidence in the conclusion, not certainty of guilt.
"""

        final_answer = self._structured_call(
            final_prompt,
            FinalInvestigatorAnswer,
        )

        valid_evidence_ids = {
            item.evidence_id
            for item in all_evidence
        }
        valid_document_ids = {
            item.document_id
            for item in all_evidence
        }

        supporting = [
            item
            for item in final_answer.supporting_evidence_ids
            if item in valid_evidence_ids
        ]

        contradicting = [
            item
            for item in final_answer.contradicting_evidence_ids
            if item in valid_evidence_ids
        ]

        documents = [
            item
            for item in final_answer.cited_document_ids
            if item in valid_document_ids
        ]

        return InvestigationResult(
            question=question,
            initial_theory=(
                initial_theory
                or trace[0].theory
            ),
            final_theory=final_answer.final_theory,
            verdict=final_answer.verdict,
            confidence=final_answer.confidence,
            reasoning_summary=(
                final_answer.reasoning_summary
            ),
            supporting_evidence_ids=supporting,
            contradicting_evidence_ids=contradicting,
            cited_document_ids=documents,
            trace=trace,
        )
