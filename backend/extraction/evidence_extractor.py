import os
import re
import time
import warnings

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.models import (
    CorpusDocument,
    DocumentExtraction,
    ExtractedClaim,
    ExtractedRelationship,
)


warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()


BANNED_RELATION_PREDICATES = {
    "expressed_gratitude_to",
    "expressed_thanks_to",
    "thanked",
    "thanked_by",
    "spoke_with",
    "observes_change_in_manner_of",
    "laughed_at",
    "listened_to",
    "conversed_with",
    "shook_hands_with",
    "greeted",
    "interacted_with",
}


def normalize_for_matching(text: str) -> str:
    if not text:
        return ""

    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "`": "'",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("_", "")
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


def validate_excerpt(source_text: str, excerpt: str) -> bool:
    if not excerpt or not excerpt.strip():
        return False

    norm_source = normalize_for_matching(source_text)
    norm_excerpt = normalize_for_matching(excerpt).strip('"\'')

    if not norm_excerpt:
        return False

    return norm_excerpt in norm_source


EXTRACTION_SYSTEM_PROMPT = """
You are an objective forensic evidence-extraction component for a detective
investigation system.

You are NOT solving the case. Extract structured case information strictly
from the supplied document chunk.

STRICT RULES

1. GROUNDING
- Extract only what is explicitly supported by the supplied text.
- Do not use outside knowledge of Sherlock Holmes or the case ending.
- Every evidence claim MUST contain a short verbatim source_excerpt copied
  directly from the supplied document.
- Every relationship MUST also contain a short verbatim source_excerpt copied
  directly from the supplied document.
- Do not paraphrase inside source_excerpt.
- Never shorten, truncate, or replace part of a source_excerpt with ellipses (...).

2. QUESTIONS ARE NOT FACTS
- Questions are not evidence that their presuppositions are true.
- Never convert an interrogative statement into an asserted fact.
- Example: "What was it you put into the wood-pile?" does NOT establish that
  the person put something there.
- If such a question is materially case-relevant, record only that the speaker
  asked the question.

3. DO NOT STRENGTHEN THEORIES
- Do not convert "X describes Y as part of a theory" into "Y definitely
  happened" or "person Z definitely performed Y".
- Attribute theories and reconstructions to their speaker.

4. EPISTEMIC STATUS
Use exactly one of: verified, unverified, disputed, misleading.

verified:
- Direct objective narration, directly observed physical evidence, or directly
  observed documentary evidence.

unverified:
- Spoken statements, witness testimony, suspect assertions, newspaper reports,
  police theories, expert opinions, deductions, recollections, accusations,
  or statements about a speaker's own investigation/knowledge unless
  independently established by objective narration.

disputed:
- The supplied text itself contains materially conflicting accounts of the
  same proposition.

misleading:
- Use only when the supplied text itself establishes that evidence or an
  apparent implication was staged, planted, contradicted, or materially
  deceptive.
- Do not label evidence misleading merely because it seems suspicious or
  because outside knowledge suggests it is false.

5. RELEVANCE
Extract only information materially useful to:
- timeline
- motive
- identity
- physical evidence
- documentary evidence
- alibi
- accusation
- contradiction
- alternative explanation
- witness reliability
- concealment or deception relevant to the case

Do not extract conversational filler, generic reactions, gratitude,
procedural chatter, or ordinary social interactions.

6. SUPPORTS / CONTRADICTS
- Use neutral targets such as "case against McFarlane",
  "McFarlane involvement", "Oldacre financial scheme",
  "case against Oldacre", or "Oldacre innocence claim".
- Do not use guilt as an established fact.
- Do not infer supports/contradicts merely from emotion, tone, demeanor,
  facial expression, or narrative implication.
- Populate these fields only when the evidentiary connection is explicit.

7. ENTITIES AND ALIASES
- Extract useful people, places, objects, events, organizations, and times.
- Never create entities from pronouns.
- Aliases may contain only proper-name variants, initials, shortened names,
  or formal-title variants that uniquely identify the same entity.
- Do not use generic descriptions such as "the victim", "the man",
  "this person", "the builder", "the accused", or pronouns as aliases.

8. RELATIONSHIPS
- Extract only case-relevant explicit relationships between entities.
- Use snake_case relationship names.
- Preserve epistemic status.
- If a relationship comes from testimony, allegation, a newspaper report,
  police theory, or expert deduction, it remains unverified unless the same
  relationship is independently established by objective narration.
- Include source_speaker when applicable.
- Never turn an accusation into a verified edge.

Return only the structured output required by the schema.
"""


class EvidenceExtractor:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ):
        resolved_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if not resolved_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY in .env."
            )

        self.client = genai.Client(api_key=resolved_key)
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or "gemini-3.5-flash-lite"
        )

    def extract_document(
        self,
        document: CorpusDocument,
        max_retries: int = 4,
    ) -> DocumentExtraction:
        user_prompt = f"""
Document ID: {document.document_id}
Source Title: {document.source_title}
Paragraphs: {document.paragraph_start}-{document.paragraph_end}

DOCUMENT TEXT:
\"\"\"
{document.text}
\"\"\"

Extract all case-relevant entities, grounded evidence claims, and grounded
relationships from this document.

The output document_id must be exactly "{document.document_id}".
"""

        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=EXTRACTION_SYSTEM_PROMPT,
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                        response_mime_type="application/json",
                        response_schema=DocumentExtraction,
                    ),
                )

                if response.parsed is not None:
                    if isinstance(response.parsed, DocumentExtraction):
                        extraction = response.parsed
                    else:
                        extraction = DocumentExtraction.model_validate(
                            response.parsed
                        )
                elif response.text:
                    extraction = DocumentExtraction.model_validate_json(
                        response.text
                    )
                else:
                    raise RuntimeError(
                        f"Empty Gemini response for {document.document_id}"
                    )

                extraction.document_id = document.document_id

                extraction.relationships = [
                    relation
                    for relation in extraction.relationships
                    if relation.relation.lower().strip()
                    not in BANNED_RELATION_PREDICATES
                ]

                extraction = self._apply_epistemic_safety_rules(
                    extraction
                )

                return self._dedupe_extraction(extraction)

            except Exception as exc:
                last_error = exc
                error_text = str(exc)

                daily_quota_exceeded = (
                    "GenerateRequestsPerDayPerProjectPerModel" in error_text
                    or "free_tier_requests" in error_text
                )

                if daily_quota_exceeded:
                    raise RuntimeError(
                        f"Daily Gemini quota exhausted for "
                        f"{self.model_name}: {exc}"
                    ) from exc

                transient = any(
                    marker in error_text
                    for marker in (
                        "503",
                        "429",
                        "UNAVAILABLE",
                        "RESOURCE_EXHAUSTED",
                    )
                )

                if transient and attempt < max_retries:
                    backoff = 2 * attempt
                    print(
                        f"    [!] Gemini transient error on "
                        f"{document.document_id}; "
                        f"retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                    continue

                raise RuntimeError(
                    f"Extraction failed for {document.document_id}: {exc}"
                ) from exc

        raise RuntimeError(
            f"Extraction failed for {document.document_id}: {last_error}"
        )

    @staticmethod
    def _apply_epistemic_safety_rules(
        extraction: DocumentExtraction,
    ) -> DocumentExtraction:
        """
        Deterministic post-processing for epistemic safety.
        """

        for claim in extraction.evidence:
            if (
                claim.source_speaker
                and claim.status == "verified"
            ):
                claim.status = "unverified"

            claim_lower = claim.claim.lower()
            excerpt_lower = claim.source_excerpt.lower()

            sensitive_words = {
                "planted",
                "fabricated",
                "staged",
                "forged",
                "faked",
            }

            for word in sensitive_words:
                if (
                    word in claim_lower
                    and word not in excerpt_lower
                ):
                    if claim.source_speaker:
                        claim.claim = (
                            f"{claim.source_speaker} states or theorizes that "
                            f"{claim.claim[0].lower()}{claim.claim[1:]}"
                        )

                    claim.status = "unverified"
                    break

        for relationship in extraction.relationships:
            if (
                relationship.source_speaker
                and relationship.status == "verified"
            ):
                relationship.status = "unverified"

        return extraction

    def validate_extraction(
        self,
        document: CorpusDocument,
        extraction: DocumentExtraction,
    ) -> tuple[DocumentExtraction, list[dict]]:
        validated_claims: list[ExtractedClaim] = []
        validated_relationships: list[ExtractedRelationship] = []
        rejected: list[dict] = []

        for claim in extraction.evidence:
            if validate_excerpt(
                document.text,
                claim.source_excerpt,
            ):
                validated_claims.append(claim)
            else:
                rejected.append(
                    {
                        "kind": "claim",
                        "document_id": document.document_id,
                        "claim": claim.claim,
                        "source_excerpt": claim.source_excerpt,
                        "reason": (
                            "Verbatim excerpt could not be matched "
                            "in source text"
                        ),
                    }
                )

        for relation in extraction.relationships:
            if validate_excerpt(
                document.text,
                relation.source_excerpt,
            ):
                validated_relationships.append(
                    relation
                )
            else:
                rejected.append(
                    {
                        "kind": "relationship",
                        "document_id": document.document_id,
                        "source_name": relation.source_name,
                        "target_name": relation.target_name,
                        "relation": relation.relation,
                        "source_excerpt": relation.source_excerpt,
                        "reason": (
                            "Verbatim excerpt could not be matched "
                            "in source text"
                        ),
                    }
                )

        extraction.evidence = validated_claims
        extraction.relationships = validated_relationships

        return extraction, rejected

    @staticmethod
    def _dedupe_extraction(
        extraction: DocumentExtraction,
    ) -> DocumentExtraction:
        seen_claims: set[tuple[str, str]] = set()
        deduped_claims = []

        for claim in extraction.evidence:
            key = (
                normalize_for_matching(claim.claim),
                normalize_for_matching(
                    claim.source_excerpt
                ),
            )

            if key not in seen_claims:
                seen_claims.add(key)
                deduped_claims.append(claim)

        seen_relationships: set[
            tuple[str, str, str, str]
        ] = set()

        deduped_relationships = []

        for relation in extraction.relationships:
            key = (
                normalize_for_matching(
                    relation.source_name
                ),
                normalize_for_matching(
                    relation.target_name
                ),
                relation.relation.lower().strip(),
                relation.status,
            )

            if key not in seen_relationships:
                seen_relationships.add(key)
                deduped_relationships.append(
                    relation
                )

        extraction.evidence = deduped_claims
        extraction.relationships = (
            deduped_relationships
        )

        return extraction
