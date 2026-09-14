import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from backend.models import (
    CaseEvidenceCorpus,
    EvidenceStatus,
    TimelineCorpus,
    TimelineEvent,
    TimelineSourceMode,
)


load_dotenv()

CANONICAL_CASE_PATH = Path("data/processed/canonical_case.json")
TIMELINE_OUTPUT_PATH = Path("data/processed/timeline_events.json")


class TimelineEventDraft(BaseModel):
    title: str
    description: str

    # Order in which this event/evidence becomes part of the investigation,
    # NOT necessarily the historical order in which the underlying event
    # originally happened.
    investigation_order: int = Field(ge=1)

    time_label: str | None = None

    # Direct evidence establishing the event itself.
    core_evidence_ids: list[str] = Field(default_factory=list)

    # Evidence that explains/interprets the event but should not determine
    # whether the event itself is verified.
    context_evidence_ids: list[str] = Field(default_factory=list)

    participant_entity_ids: list[str] = Field(default_factory=list)
    location_entity_ids: list[str] = Field(default_factory=list)


class TimelineDraft(BaseModel):
    events: list[TimelineEventDraft] = Field(default_factory=list)


def event_status(
    statuses: list[EvidenceStatus],
) -> EvidenceStatus:
    """
    Derive event status from CORE evidence only.

    Because core evidence is defined as evidence that directly establishes the
    event, one verified direct item is enough to establish that the event
    occurred, even when contextual testimony remains unverified.
    """
    if any(status == "verified" for status in statuses):
        return "verified"

    if any(status == "disputed" for status in statuses):
        return "disputed"

    if any(status == "misleading" for status in statuses):
        return "misleading"

    return "unverified"


def source_mode(cited_evidence) -> TimelineSourceMode:
    """
    Deterministically classify how the core event reaches the record.
    """
    if any(
        item.status == "verified"
        and item.source_speaker is None
        for item in cited_evidence
    ):
        return "observed"

    if cited_evidence and all(
        item.source_speaker is not None
        for item in cited_evidence
    ):
        return "recounted"

    return "mixed"


def main() -> None:
    if not CANONICAL_CASE_PATH.exists():
        raise FileNotFoundError(
            "Canonical case data not found. Run canonicalization first."
        )

    with CANONICAL_CASE_PATH.open("r", encoding="utf-8") as file:
        case = CaseEvidenceCorpus.model_validate(json.load(file))

    evidence_by_id = {
        item.evidence_id: item
        for item in case.evidence
    }
    entity_by_id = {
        item.entity_id: item
        for item in case.entities
    }

    evidence_payload = [
        {
            "evidence_id": item.evidence_id,
            "document_id": item.document_id,
            "claim": item.claim,
            "status": item.status,
            "evidence_type": item.evidence_type,
            "source_speaker": item.source_speaker,
            "entity_ids": item.entity_ids,
            "entity_names": item.entity_names,
        }
        for item in case.evidence
    ]

    entity_payload = [
        {
            "entity_id": item.entity_id,
            "name": item.canonical_name,
            "type": item.entity_type,
        }
        for item in case.entities
    ]

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key:
        raise ValueError("Gemini API key not found.")

    model_name = (
        os.getenv("GEMINI_MODEL")
        or "gemini-3.5-flash-lite"
    )

    client = genai.Client(api_key=api_key)

    prompt = f"""
Build a compact INVESTIGATION TIMELINE from the supplied canonical EVIDENCE and
ENTITY tables.

This is NOT a claim that every event can be placed on one perfect historical
clock. The timeline's primary sequence is the order in which major facts,
discoveries, testimony, and observed events become relevant to the
investigation.

Important rules:
- Return 8 to 14 major investigation events.
- Every event MUST contain at least one CORE evidence ID.
- Use ONLY supplied EV_### and ENT_### IDs.
- Do not use outside Sherlock Holmes knowledge.

CORE vs CONTEXT evidence:
- `core_evidence_ids` = evidence that DIRECTLY establishes the event described
  in the title/description.
- `context_evidence_ids` = evidence that helps explain motive, interpretation,
  consequences, or surrounding circumstances but does not itself establish
  that the event happened.
- Do not put an item in context merely because it is unverified. A witness
  statement can be core evidence for the event "Witness states X", because
  the occurrence being represented is the statement itself.
- Keep these sets small and precise.

Ordering:
- `investigation_order` is the order in which the event/fact enters the
  investigation or is discovered/presented.
- Do NOT reorder a later recounting simply because the event being recounted
  originally happened earlier.
- Example: a suspect recounting an earlier meeting occurs later in the
  investigation sequence even if `time_label` says "previous evening".

Time labels:
- `time_label` is a short source-supported relative phrase such as
  "previous evening", "night of the incident", "next morning", or
  "during the search".
- If no defensible relative time phrase exists, use null.
- Never invent dates or clock times.

Descriptions:
- `description` must be a concise synthesis of CORE evidence.
- Do not turn an allegation, theory, or testimony into objective fact.
- A directly narrated/observed physical event should be described plainly.

Entities:
- `participant_entity_ids` should include people/objects/organizations directly
  involved when supported.
- `location_entity_ids` must contain only place entity IDs.

Event status and source mode are NOT chosen by you. Python derives them later
from CORE evidence only.

ENTITIES:
{json.dumps(entity_payload, indent=2, ensure_ascii=False)}

EVIDENCE:
{json.dumps(evidence_payload, indent=2, ensure_ascii=False)}
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
            response_schema=TimelineDraft,
        ),
    )

    if response.parsed is not None:
        if isinstance(response.parsed, TimelineDraft):
            draft = response.parsed
        else:
            draft = TimelineDraft.model_validate(response.parsed)
    else:
        draft = TimelineDraft.model_validate_json(response.text)

    valid_events: list[TimelineEventDraft] = []
    seen_signatures: set[
        tuple[str, tuple[str, ...]]
    ] = set()

    for event in draft.events:
        core_ids = list(
            dict.fromkeys(
                evidence_id
                for evidence_id in event.core_evidence_ids
                if evidence_id in evidence_by_id
            )
        )

        if not core_ids:
            continue

        context_ids = list(
            dict.fromkeys(
                evidence_id
                for evidence_id in event.context_evidence_ids
                if (
                    evidence_id in evidence_by_id
                    and evidence_id not in core_ids
                )
            )
        )

        participant_ids = list(
            dict.fromkeys(
                entity_id
                for entity_id in event.participant_entity_ids
                if entity_id in entity_by_id
            )
        )

        location_ids = list(
            dict.fromkeys(
                entity_id
                for entity_id in event.location_entity_ids
                if (
                    entity_id in entity_by_id
                    and entity_by_id[entity_id].entity_type == "place"
                )
            )
        )

        signature = (
            event.title.strip().lower(),
            tuple(sorted(core_ids)),
        )
        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)

        valid_events.append(
            event.model_copy(
                update={
                    "core_evidence_ids": core_ids,
                    "context_evidence_ids": context_ids,
                    "participant_entity_ids": participant_ids,
                    "location_entity_ids": location_ids,
                }
            )
        )

    valid_events.sort(
        key=lambda item: (
            item.investigation_order,
            item.title.lower(),
        )
    )

    timeline_events: list[TimelineEvent] = []

    for index, event in enumerate(valid_events, start=1):
        core_evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in event.core_evidence_ids
        ]

        all_evidence_ids = list(
            dict.fromkeys(
                event.core_evidence_ids
                + event.context_evidence_ids
            )
        )

        document_ids = sorted(
            {
                evidence_by_id[evidence_id].document_id
                for evidence_id in all_evidence_ids
            }
        )

        timeline_events.append(
            TimelineEvent(
                event_id=f"EVENT_{index:03d}",
                title=event.title.strip(),
                description=event.description.strip(),
                investigation_order=index,
                time_label=(
                    event.time_label.strip()
                    if event.time_label
                    and event.time_label.strip()
                    else None
                ),
                status=event_status(
                    [item.status for item in core_evidence]
                ),
                source_mode=source_mode(core_evidence),
                core_evidence_ids=event.core_evidence_ids,
                context_evidence_ids=event.context_evidence_ids,
                participant_entity_ids=event.participant_entity_ids,
                location_entity_ids=event.location_entity_ids,
                document_ids=document_ids,
            )
        )

    if len(timeline_events) < 6:
        raise RuntimeError(
            "Timeline extraction returned too few grounded events. "
            "Review the model output before using it."
        )

    timeline = TimelineCorpus(
        case_id=case.case_id,
        events=timeline_events,
    )

    TIMELINE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TIMELINE_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            timeline.model_dump(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print("CASEFILE AI — INVESTIGATION TIMELINE BUILD")
    print("=" * 72)
    print(f"Events: {len(timeline.events)}")
    print()

    for event in timeline.events:
        print(
            f"{event.event_id} | {event.status.upper():10} | "
            f"{event.source_mode.upper():9} | "
            f"{event.time_label or 'time unspecified'}"
        )
        print(f"  {event.title}")
        print(
            "  Core: "
            + ", ".join(event.core_evidence_ids)
        )
        if event.context_evidence_ids:
            print(
                "  Context: "
                + ", ".join(event.context_evidence_ids)
            )
        print()

    print(f"Saved: {TIMELINE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
