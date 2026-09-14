import json
import os
from collections import OrderedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.agents.evidence_store import AgentEvidenceStore
from backend.agents.schemas import (
    CoveragePlan,
    FinalInvestigatorAnswer,
    InvestigationResult,
    InvestigatorStep,
    SufficiencyDecision,
)


load_dotenv()


INVESTIGATOR_SYSTEM_PROMPT = """
You are the Investigator Agent in CaseFile AI.

You must reason only from supplied retrieved evidence.

Rules:
- Treat verified evidence as stronger than unverified testimony.
- Never convert unverified testimony into established fact.
- Explicitly consider evidence that conflicts with the current theory.
- If evidence is insufficient, identify the exact missing fact and request a
  focused retrieval query.
- Do not invent facts, evidence IDs, document IDs, people, events, or motives.
- Every factual conclusion in the final answer must be grounded in supplied
  evidence IDs and document IDs.
- A theory may be plausible without being proven.
- Prefer direct verified observations when they resolve an uncertainty that is
  otherwise supported only by testimony or theory.
- Do not stop merely because several pieces of evidence point in one direction;
  actively look for decisive evidence that could resolve the remaining gap.
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
            raise ValueError("Gemini API key not found.")

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
            return schema.model_validate(response.parsed)

        return schema.model_validate_json(response.text)

    @staticmethod
    def _format_evidence(evidence) -> str:
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

    def _assess_sufficiency(
        self,
        question: str,
        theory: str,
        evidence_text: str,
        coverage_phase: bool = False,
    ) -> SufficiencyDecision:
        phase_instruction = (
            """
This is the final coverage sweep. You have already had multiple retrieval
attempts. Carefully check whether the newly accumulated evidence directly
resolves the previously missing fact. Give special weight to verified direct
observations. Do not preserve an earlier uncertainty if the evidence now
directly resolves it.

Important generic rule:
- if the answer says it is unresolved whether a person was alive or dead at a
  relevant time, and VERIFIED evidence directly observes that same person
  physically acting or appearing at that time, the alive/dead uncertainty is
  resolved for that observation;
- similarly, any VERIFIED direct observation that explicitly resolves the
  stated gap must override an earlier inference based only on testimony,
  theory, absence of remains, or circumstantial evidence.
"""
            if coverage_phase
            else ""
        )

        prompt = f"""
QUESTION:
{question}

CURRENT WORKING THEORY:
{theory}

ACCUMULATED EVIDENCE:
{evidence_text}

{phase_instruction}

Evaluate whether the accumulated evidence is sufficient to answer the question
responsibly.

You must revise the working theory based on the evidence, even if the evidence
is insufficient.

If insufficient:
- explain the precise unresolved evidence gap;
- produce one focused next_query designed to retrieve missing or adversarial
  evidence.

If sufficient:
- next_query must be null.
"""

        return self._structured_call(
            prompt,
            SufficiencyDecision,
        )

    def _build_coverage_plan(
        self,
        question: str,
        theory: str,
        unresolved_gap: str,
        evidence_text: str,
    ) -> CoveragePlan:
        prompt = f"""
QUESTION:
{question}

CURRENT THEORY:
{theory}

UNRESOLVED GAP:
{unresolved_gap}

EVIDENCE ALREADY SEEN:
{evidence_text}

Create 2 to 4 short retrieval queries whose only purpose is to determine
whether decisive evidence was missed.

The queries must be complementary:
1. seek a direct verified observation that resolves the gap;
2. seek evidence contradicting the current theory;
3. seek timeline/event evidence that could change the answer;
4. optionally seek a relevant alternative explanation.

Do not assume which side is correct. Write search-style queries, not prose
answers.
"""

        return self._structured_call(
            prompt,
            CoveragePlan,
        )

    def investigate(
        self,
        question: str,
        initial_theory: str | None = None,
        max_iterations: int = 3,
        top_k: int = 8,
    ) -> InvestigationResult:
        theory = (
            initial_theory
            or (
                "No prior theory supplied. Form a cautious working theory "
                "from the retrieved evidence."
            )
        )

        query = question
        trace: list[InvestigatorStep] = []
        seen_evidence = OrderedDict()

        last_decision: SufficiencyDecision | None = None
        termination_reason = "retry_limit"
        coverage_sweep_used = False
        coverage_queries: list[str] = []

        for iteration in range(1, max_iterations + 1):
            retrieved = self.store.search(
                query=query,
                top_k=top_k,
            )

            for item in retrieved:
                seen_evidence[item.evidence_id] = item

            evidence_text = self._format_evidence(
                list(seen_evidence.values())
            )

            decision = self._assess_sufficiency(
                question=question,
                theory=theory,
                evidence_text=evidence_text,
            )
            last_decision = decision

            trace.append(
                InvestigatorStep(
                    iteration=iteration,
                    phase="investigation",
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
                termination_reason = "sufficient"
                break

            if not decision.next_query:
                termination_reason = "no_next_query"
                break

            query = decision.next_query

        # -----------------------------------------------------------------
        # Coverage sweep
        #
        # A grounded answer can still be incomplete if retrieval missed a
        # decisive item. Before accepting a retry-limit/no-query termination,
        # generate several gap-directed queries and perform a broader
        # multi-query retrieval pass, then re-evaluate sufficiency once.
        # -----------------------------------------------------------------
        if (
            last_decision is not None
            and not last_decision.sufficient
        ):
            coverage_sweep_used = True

            existing_evidence_text = self._format_evidence(
                list(seen_evidence.values())
            )

            plan = self._build_coverage_plan(
                question=question,
                theory=theory,
                unresolved_gap=last_decision.reason,
                evidence_text=existing_evidence_text,
            )

            model_queries = [
                item.strip()
                for item in plan.queries
                if item and item.strip()
            ][:4]

            deterministic_queries = [
                f"direct verified observation resolving: {last_decision.reason}",
                f"decisive evidence contradicting current theory: {theory}",
                f"direct observed event relevant to: {question}",
            ]

            coverage_queries = list(
                dict.fromkeys(
                    model_queries + deterministic_queries
                )
            )

            coverage_results = self.store.search_many(
                coverage_queries,
                top_k=max(14, top_k * 2),
                per_query_k=max(12, top_k * 2),
            )

            # Completeness guard:
            # this case contains only a small verified subset, so before
            # declaring a material question unresolved we inspect ALL verified
            # evidence. This catches decisive direct observations that a
            # query-based retriever may miss.
            verified_safety_pass = self.store.verified_evidence()

            for item in coverage_results:
                seen_evidence[item.evidence_id] = item

            for item in verified_safety_pass:
                seen_evidence[item.evidence_id] = item

            accumulated_evidence = list(seen_evidence.values())
            accumulated_text = self._format_evidence(
                accumulated_evidence
            )

            coverage_decision = self._assess_sufficiency(
                question=question,
                theory=theory,
                evidence_text=accumulated_text,
                coverage_phase=True,
            )

            trace.append(
                InvestigatorStep(
                    iteration=len(trace) + 1,
                    phase="coverage_sweep",
                    query=" | ".join(coverage_queries),
                    theory=theory,
                    sufficient=coverage_decision.sufficient,
                    reason=coverage_decision.reason,
                    next_query=None,
                    evidence_ids=list(
                        dict.fromkeys(
                            [
                                item.evidence_id
                                for item in coverage_results
                            ]
                            + [
                                item.evidence_id
                                for item in verified_safety_pass
                            ]
                        )
                    ),
                    document_ids=sorted(
                        {
                            item.document_id
                            for item in (
                                coverage_results
                                + verified_safety_pass
                            )
                        }
                    ),
                )
            )

            last_decision = coverage_decision
            theory = coverage_decision.revised_theory

            if coverage_decision.sufficient:
                termination_reason = "sufficient_after_coverage"
            else:
                termination_reason = "retry_limit"

        needs_more_evidence = not (
            last_decision is not None
            and last_decision.sufficient
        )

        all_evidence = list(seen_evidence.values())
        evidence_text = self._format_evidence(all_evidence)

        final_prompt = f"""
QUESTION:
{question}

INITIAL INPUT THEORY:
{initial_theory or "None supplied"}

FINAL WORKING THEORY:
{theory}

TERMINATION:
{termination_reason}

EVIDENCE STILL INSUFFICIENT:
{needs_more_evidence}

COVERAGE SWEEP USED:
{coverage_sweep_used}

COVERAGE QUERIES:
{json.dumps(coverage_queries, indent=2)}

AGENT TRACE:
{json.dumps([step.model_dump() for step in trace], indent=2)}

AVAILABLE EVIDENCE:
{evidence_text}

Return the final grounded investigation result.

Requirements:
- verdict must be concise and cautious;
- reasoning_summary must distinguish verified facts from unverified claims;
- direct verified observations should override an earlier unresolved uncertainty
  when they actually resolve it;
- if EVIDENCE STILL INSUFFICIENT is true, explicitly state the unresolved gap;
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
                or "No prior user theory supplied."
            ),
            final_theory=final_answer.final_theory,
            verdict=final_answer.verdict,
            confidence=final_answer.confidence,
            reasoning_summary=final_answer.reasoning_summary,
            supporting_evidence_ids=supporting,
            contradicting_evidence_ids=contradicting,
            cited_document_ids=documents,
            needs_more_evidence=needs_more_evidence,
            termination_reason=termination_reason,
            coverage_sweep_used=coverage_sweep_used,
            coverage_queries=coverage_queries,
            trace=trace,
        )
