from typing import Literal

from pydantic import BaseModel, Field


EvidenceStatus = Literal[
    "verified",
    "unverified",
    "disputed",
    "misleading",
]

EntityType = Literal[
    "person",
    "place",
    "object",
    "event",
    "organization",
    "time",
    "other",
]

EvidenceType = Literal[
    "testimony",
    "physical",
    "documentary",
    "timeline",
    "motive",
    "observation",
    "background",
    "other",
]


# =====================================================================
# RETRIEVAL MODELS
# =====================================================================

class CorpusDocument(BaseModel):
    document_id: str
    case_id: str
    source_title: str
    source_url: str
    paragraph_start: int
    paragraph_end: int
    word_count: int
    text: str


class SearchResult(BaseModel):
    document_id: str
    score: float
    text: str
    source_title: str
    paragraph_start: int
    paragraph_end: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


# =====================================================================
# LAYER 1: RAW LLM EXTRACTION SCHEMAS
# Document-local; Gemini does not assign permanent IDs.
# =====================================================================

class ExtractedEntity(BaseModel):
    name: str = Field(
        description="Primary proper name of the entity in this document."
    )
    entity_type: EntityType
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Proper-name variants, initials, shortened names, or formal-title "
            "variants that uniquely refer to the same entity."
        ),
    )
    role: str | None = Field(
        default=None,
        description="Brief case-relevant role in this document.",
    )


class ExtractedClaim(BaseModel):
    claim: str = Field(
        description=(
            "Concise, self-contained case-relevant proposition or observation."
        )
    )
    evidence_type: EvidenceType
    status: EvidenceStatus = Field(
        description=(
            "Epistemic status: verified for directly observed/narrated facts; "
            "unverified for testimony, allegations, theories, or expert opinions; "
            "disputed for explicitly conflicting accounts; misleading only when "
            "the supplied text itself establishes that the evidence or impression "
            "was staged, planted, contradicted, or materially deceptive."
        )
    )
    source_speaker: str | None = Field(
        default=None,
        description=(
            "Speaker/source of testimony, or null for objective narration."
        ),
    )
    source_excerpt: str = Field(
        description=(
            "Short verbatim excerpt copied from the source document that directly "
            "supports this claim."
        )
    )
    entity_names: list[str] = Field(default_factory=list)
    supports: list[str] = Field(
        default_factory=list,
        description=(
            "Neutral evidentiary hypotheses explicitly supported by this claim, "
            "for example 'case against McFarlane' or 'Oldacre financial scheme'."
        ),
    )
    contradicts: list[str] = Field(
        default_factory=list,
        description=(
            "Neutral hypotheses or claims explicitly challenged by this evidence."
        ),
    )


class ExtractedRelationship(BaseModel):
    source_name: str
    target_name: str
    relation: str = Field(
        description="Case-relevant relationship predicate in snake_case."
    )
    status: EvidenceStatus = Field(
        default="unverified",
        description=(
            "Epistemic status of the relationship. Testimony, allegation, police "
            "theory, or expert deduction remains unverified unless independently "
            "established by objective narration."
        ),
    )
    source_speaker: str | None = None
    source_excerpt: str = Field(
        description=(
            "Short verbatim excerpt copied from the source document that directly "
            "supports this relationship."
        )
    )


class DocumentExtraction(BaseModel):
    document_id: str
    entities: list[ExtractedEntity] = Field(default_factory=list)
    evidence: list[ExtractedClaim] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# =====================================================================
# LAYER 2: NORMALIZED CANONICAL MODELS
# Cross-corpus; stable IDs are assigned by Python canonicalization.
# =====================================================================

class CanonicalEntity(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    mention_count: int = 0
    description: str | None = None


class EvidenceClaim(BaseModel):
    evidence_id: str
    document_id: str
    claim: str
    evidence_type: EvidenceType
    status: EvidenceStatus
    source_speaker: str | None = None
    source_excerpt: str
    is_grounded: bool = True
    entity_ids: list[str] = Field(default_factory=list)
    entity_names: list[str] = Field(default_factory=list)
    supports: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)


class EntityRelationship(BaseModel):
    relationship_id: str
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    relation: str
    status: EvidenceStatus = "unverified"
    source_speaker: str | None = None
    source_excerpt: str | None = None
    is_grounded: bool = True
    document_id: str | None = None


# Backward-compatibility aliases.
Entity = CanonicalEntity
Relationship = EntityRelationship


class EvidenceBundle(BaseModel):
    document_id: str
    entities: list[CanonicalEntity] = Field(default_factory=list)
    evidence: list[EvidenceClaim] = Field(default_factory=list)
    relationships: list[EntityRelationship] = Field(default_factory=list)


class CaseEvidenceCorpus(BaseModel):
    case_id: str
    entities: list[CanonicalEntity] = Field(default_factory=list)
    evidence: list[EvidenceClaim] = Field(default_factory=list)
    relationships: list[EntityRelationship] = Field(default_factory=list)

# =====================================================================
# TIMELINE / EVENT MODELS
# =====================================================================

TimelineSourceMode = Literal[
    "observed",
    "recounted",
    "mixed",
]


class TimelineEvent(BaseModel):
    event_id: str
    title: str
    description: str

    # Order in which the event/evidence enters the investigation. This is
    # deliberately separate from the underlying story-time label.
    investigation_order: int = Field(ge=1)

    # Relative time phrase supported by the source, e.g. "night of the
    # incident" or "next morning". Null if the source does not support one.
    time_label: str | None = None

    # Determined from CORE evidence only.
    status: EvidenceStatus
    source_mode: TimelineSourceMode

    # Evidence that directly establishes the event itself.
    core_evidence_ids: list[str] = Field(default_factory=list)

    # Related evidence that explains motive, interpretation, or surrounding
    # context, but is not used to determine whether the event occurred.
    context_evidence_ids: list[str] = Field(default_factory=list)

    participant_entity_ids: list[str] = Field(default_factory=list)
    location_entity_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)

    @property
    def evidence_ids(self) -> list[str]:
        """Backward-compatible combined evidence view."""
        return list(
            dict.fromkeys(
                self.core_evidence_ids
                + self.context_evidence_ids
            )
        )


class TimelineCorpus(BaseModel):
    case_id: str
    timeline_type: Literal["investigation_timeline"] = "investigation_timeline"
    events: list[TimelineEvent] = Field(default_factory=list)
