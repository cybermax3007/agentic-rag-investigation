import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.agents.evidence_store import AgentEvidenceStore
from backend.agents.schemas import FactCheckResult, InvestigationResult


load_dotenv()


FACT_CHECKER_SYSTEM_PROMPT = """
You are the adversarial Fact-Checker in CaseFile AI.

Your job is NOT to agree with the Investigator.

You must:
- search for evidence that weakens the Investigator's theory;
- identify conflicting timelines;
- identify alternative explanations;
- identify evidence supporting other suspects or interpretations;
- flag reliance on unverified testimony;
- distinguish absence of evidence from evidence of absence.

Never invent evidence IDs or document IDs.
Use only the supplied retrieved evidence.
"""


class FactCheckerAgent:
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

    def fact_check(
        self,
        investigation: InvestigationResult,
        top_k: int = 10,
    ) -> FactCheckResult:
        theory = investigation.final_theory

        adversarial_queries = [
            (
                "contradictions counter-evidence weaknesses against theory: "
                f"{theory}"
            ),
            (
                "alternative explanation staged disappearance framing "
                f"against theory: {theory}"
            ),
            (
                "timeline conflict alive body remains hidden room "
                f"against theory: {theory}"
            ),
            (
                "evidence supporting another candidate or interpretation "
                f"against theory: {theory}"
            ),
        ]

        evidence = self.store.search_many(
            adversarial_queries,
            top_k=top_k,
            per_query_k=max(8, top_k),
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

        prompt = f"""
INVESTIGATOR RESULT:
{json.dumps(investigation.model_dump(), indent=2)}

ADVERSARIAL SEARCH INTENTS:
{json.dumps(adversarial_queries, indent=2)}

INDEPENDENT ADVERSARIAL RETRIEVAL:
{evidence_text}

Attack the Investigator's conclusion using only the supplied evidence.

Specifically check:
1. What evidence is weak, testimonial, or inferential?
2. What alternative explanations remain?
3. Are there conflicting or incomplete timeline points?
4. Does any evidence support another suspect or interpretation?
5. Which Investigator evidence IDs deserve challenge?

Return a concise adversarial assessment.
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=FACT_CHECKER_SYSTEM_PROMPT,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(
                    thinking_level="minimal"
                ),
                response_mime_type="application/json",
                response_schema=FactCheckResult,
            ),
        )

        if response.parsed is not None:
            if isinstance(response.parsed, FactCheckResult):
                result = response.parsed
            else:
                result = FactCheckResult.model_validate(
                    response.parsed
                )
        else:
            result = FactCheckResult.model_validate_json(
                response.text
            )

        valid_evidence_ids = {
            item.evidence_id
            for item in evidence
        } | set(
            investigation.supporting_evidence_ids
        ) | set(
            investigation.contradicting_evidence_ids
        )

        valid_document_ids = {
            item.document_id
            for item in evidence
        } | set(
            investigation.cited_document_ids
        )

        result.challenged_evidence_ids = [
            item
            for item in result.challenged_evidence_ids
            if item in valid_evidence_ids
        ]

        result.evidence_for_alternatives = [
            item
            for item in result.evidence_for_alternatives
            if item in valid_evidence_ids
        ]

        result.cited_document_ids = [
            item
            for item in result.cited_document_ids
            if item in valid_document_ids
        ]

        return result
