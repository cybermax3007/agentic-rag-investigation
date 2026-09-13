import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from backend.models import (
    CanonicalEntity,
    CaseEvidenceCorpus,
    DocumentExtraction,
    EntityRelationship,
    EvidenceClaim,
)


RAW_EXTRACTIONS_PATH = Path(
    "data/processed/raw_extractions.json"
)
CANONICAL_OUTPUT_PATH = Path(
    "data/processed/canonical_case.json"
)
CANONICAL_AUDIT_PATH = Path(
    "data/processed/canonicalization_audit.json"
)


# ---------------------------------------------------------------------
# Small, transparent case-specific identity map.
# We prefer explicit identity rules over fuzzy matching because a false
# merge is more damaging than leaving two entities separate.
# ---------------------------------------------------------------------

CASE_CANONICAL_OVERRIDES = {
    # Sherlock Holmes
    "holmes": "Sherlock Holmes",
    "mr holmes": "Sherlock Holmes",
    "sherlock holmes": "Sherlock Holmes",

    # Watson
    "watson": "Watson",
    "john watson": "Watson",
    "dr watson": "Watson",
    "doctor watson": "Watson",

    # John Hector McFarlane
    "mcfarlane": "John Hector McFarlane",
    "mr mcfarlane": "John Hector McFarlane",
    "john mcfarlane": "John Hector McFarlane",
    "john hector mcfarlane": "John Hector McFarlane",
    "mr john hector mcfarlane": "John Hector McFarlane",

    # McFarlane's mother — MUST remain distinct from John McFarlane.
    "mrs mcfarlane": "Mrs. McFarlane",
    "mcfarlane's mother": "Mrs. McFarlane",
    "mcfarlanes mother": "Mrs. McFarlane",

    # Jonas Oldacre
    "oldacre": "Jonas Oldacre",
    "mr oldacre": "Jonas Oldacre",
    "jonas oldacre": "Jonas Oldacre",
    "mr jonas oldacre": "Jonas Oldacre",

    # Lestrade
    "lestrade": "Inspector Lestrade",
    "inspector lestrade": "Inspector Lestrade",
}


KNOWN_PERSON_CANONICALS = {
    "Sherlock Holmes",
    "Watson",
    "John Hector McFarlane",
    "Mrs. McFarlane",
    "Jonas Oldacre",
    "Inspector Lestrade",
}


TYPE_PRIORITY = {
    "person": 0,
    "place": 1,
    "organization": 2,
    "object": 3,
    "event": 4,
    "time": 5,
    "other": 6,
}


# Descriptive phrases are NOT stable aliases.
GENERIC_ALIAS_EXACT = {
    "our client",
    "the prisoner",
    "the young man",
    "this young man",
    "unfortunate youngster",
    "the victim",
    "the man",
    "this person",
    "the builder",
    "the accused",
    "master",
    "the fellow",
    "that rat",
    "the older man",
    "the father",
    "the mother",
    "my friend",
}

GENERIC_ALIAS_PREFIXES = (
    "the ",
    "this ",
    "our ",
    "unfortunate ",
    "late lamented ",
    "young ",
    "poor ",
    "little ",
)


def normalize_surface(name: str) -> str:
    """
    Conservative normalization for identity comparison.

    Important: we DO NOT strip titles generically. Doing so caused
    'Mrs. McFarlane' to collapse into John Hector McFarlane.
    """
    text = name.strip().lower()

    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    text = re.sub(r"[^\w\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def identity_key(name: str) -> str:
    normalized = normalize_surface(name)

    if normalized in CASE_CANONICAL_OVERRIDES:
        return normalize_surface(
            CASE_CANONICAL_OVERRIDES[normalized]
        )

    return normalized


def preferred_override(name: str) -> str | None:
    return CASE_CANONICAL_OVERRIDES.get(
        normalize_surface(name)
    )


def is_good_alias(alias: str) -> bool:
    """
    Keep only stable proper-name/title variants.

    We intentionally reject descriptive phrases even when Gemini supplied
    them as aliases, because they can incorrectly merge unrelated people.
    """
    if not alias or not alias.strip():
        return False

    normalized = normalize_surface(alias)

    if normalized in GENERIC_ALIAS_EXACT:
        return False

    if normalized.startswith(GENERIC_ALIAS_PREFIXES):
        return False

    # Descriptive comma-heavy phrases are generally not proper aliases.
    if alias.count(",") >= 2:
        return False

    return True


def infer_missing_type(name: str) -> str:
    override = preferred_override(name)

    if override in KNOWN_PERSON_CANONICALS:
        return "person"

    return "other"


def display_score(name: str) -> tuple[int, int]:
    """
    Prefer fuller proper names over short forms.
    """
    normalized = normalize_surface(name)
    return (
        len(normalized.split()),
        len(normalized),
    )


class EntityAccumulator:
    def __init__(self):
        self.names: set[str] = set()
        self.primary_names: set[str] = set()
        self.document_ids: set[str] = set()
        self.type_counts: Counter = Counter()
        self.roles: set[str] = set()
        self.mention_count = 0
        self.preferred_name: str | None = None

    def add(
        self,
        name: str,
        entity_type: str,
        document_id: str,
        *,
        is_primary: bool,
        aliases: list[str] | None = None,
        role: str | None = None,
    ) -> None:
        if not name or not name.strip():
            return

        clean_name = name.strip()

        self.names.add(clean_name)
        self.document_ids.add(document_id)
        self.type_counts[entity_type] += 1

        if is_primary:
            self.primary_names.add(clean_name)
            self.mention_count += 1

        if aliases:
            for alias in aliases:
                if is_good_alias(alias):
                    self.names.add(alias.strip())

        if role and role.strip():
            self.roles.add(role.strip())

        override = preferred_override(clean_name)
        if override:
            self.preferred_name = override

    def resolved_type(self) -> str:
        if self.preferred_name in KNOWN_PERSON_CANONICALS:
            return "person"

        if not self.type_counts:
            return "other"

        ranked = sorted(
            self.type_counts.items(),
            key=lambda item: (
                -item[1],
                TYPE_PRIORITY.get(item[0], 99),
                item[0],
            ),
        )

        return ranked[0][0]

    def resolved_name(self) -> str:
        if self.preferred_name:
            return self.preferred_name

        candidates = (
            self.primary_names
            if self.primary_names
            else self.names
        )

        if not candidates:
            return "Unknown Entity"

        return max(
            candidates,
            key=display_score,
        )

    def description(self) -> str | None:
        if not self.roles:
            return None

        return "; ".join(sorted(self.roles))


class EvidenceCanonicalizer:
    def __init__(
        self,
        raw_path: Path = RAW_EXTRACTIONS_PATH,
    ):
        self.raw_path = raw_path

    def load_raw(self) -> list[DocumentExtraction]:
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Raw extraction file not found: {self.raw_path}"
            )

        with self.raw_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw = json.load(file)

        extractions = [
            DocumentExtraction.model_validate(item)
            for item in raw
        ]

        extractions.sort(
            key=lambda item: item.document_id
        )

        return extractions

    def canonicalize(
        self,
    ) -> tuple[CaseEvidenceCorpus, dict]:
        extractions = self.load_raw()

        accumulators: dict[
            str,
            EntityAccumulator,
        ] = defaultdict(EntityAccumulator)

        rejected_aliases: list[dict] = []

        # -------------------------------------------------------------
        # Pass 1: explicit extracted entities.
        # -------------------------------------------------------------
        for extraction in extractions:
            for entity in extraction.entities:
                key = identity_key(entity.name)

                good_aliases = []

                for alias in entity.aliases:
                    if is_good_alias(alias):
                        good_aliases.append(alias)
                    else:
                        rejected_aliases.append(
                            {
                                "document_id": extraction.document_id,
                                "entity_name": entity.name,
                                "rejected_alias": alias,
                            }
                        )

                accumulators[key].add(
                    entity.name,
                    entity.entity_type,
                    extraction.document_id,
                    is_primary=True,
                    aliases=good_aliases,
                    role=entity.role,
                )

                for alias in good_aliases:
                    # Alias is attached to the SAME canonical cluster.
                    accumulators[key].add(
                        alias,
                        entity.entity_type,
                        extraction.document_id,
                        is_primary=False,
                    )

        surface_to_key: dict[str, str] = {}

        for key, accumulator in accumulators.items():
            surface_to_key[key] = key

            for name in accumulator.names:
                surface_to_key[
                    normalize_surface(name)
                ] = key

                surface_to_key[
                    identity_key(name)
                ] = key

        inferred_mentions = 0

        def ensure_entity(
            name: str,
            document_id: str,
        ) -> str:
            nonlocal inferred_mentions

            normalized = normalize_surface(name)
            resolved = identity_key(name)

            if normalized in surface_to_key:
                key = surface_to_key[normalized]
            elif resolved in surface_to_key:
                key = surface_to_key[resolved]
            else:
                key = resolved
                inferred_mentions += 1

                accumulators[key].add(
                    name,
                    infer_missing_type(name),
                    document_id,
                    is_primary=False,
                )

                surface_to_key[normalized] = key
                surface_to_key[resolved] = key

            accumulators[key].document_ids.add(
                document_id
            )

            return key

        # -------------------------------------------------------------
        # Pass 2: resolve claim references and relationship endpoints.
        # -------------------------------------------------------------
        for extraction in extractions:
            for claim in extraction.evidence:
                for name in claim.entity_names:
                    ensure_entity(
                        name,
                        extraction.document_id,
                    )

            for relation in extraction.relationships:
                ensure_entity(
                    relation.source_name,
                    extraction.document_id,
                )
                ensure_entity(
                    relation.target_name,
                    extraction.document_id,
                )

        # -------------------------------------------------------------
        # Stable canonical entity IDs.
        # -------------------------------------------------------------
        sorted_entity_keys = sorted(
            accumulators.keys(),
            key=lambda key: (
                TYPE_PRIORITY.get(
                    accumulators[key].resolved_type(),
                    99,
                ),
                accumulators[key].resolved_name().lower(),
                key,
            ),
        )

        entity_id_by_key: dict[str, str] = {}
        canonical_entities: list[
            CanonicalEntity
        ] = []

        for index, key in enumerate(
            sorted_entity_keys,
            start=1,
        ):
            accumulator = accumulators[key]

            entity_id = f"ENT_{index:03d}"
            entity_id_by_key[key] = entity_id

            canonical_name = (
                accumulator.resolved_name()
            )

            aliases = sorted(
                {
                    name
                    for name in accumulator.names
                    if (
                        is_good_alias(name)
                        and normalize_surface(name)
                        != normalize_surface(canonical_name)
                    )
                },
                key=str.lower,
            )

            canonical_entities.append(
                CanonicalEntity(
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    entity_type=(
                        accumulator.resolved_type()
                    ),
                    aliases=aliases,
                    document_ids=sorted(
                        accumulator.document_ids
                    ),
                    mention_count=(
                        accumulator.mention_count
                    ),
                    description=(
                        accumulator.description()
                    ),
                )
            )

        def resolve_id(
            name: str,
            document_id: str,
        ) -> str:
            normalized = normalize_surface(name)
            resolved = identity_key(name)

            key = (
                surface_to_key.get(normalized)
                or surface_to_key.get(resolved)
                or resolved
            )

            if key not in entity_id_by_key:
                raise KeyError(
                    f"Unresolved entity '{name}' "
                    f"in {document_id}"
                )

            return entity_id_by_key[key]

        # -------------------------------------------------------------
        # Stable evidence IDs.
        # -------------------------------------------------------------
        canonical_evidence: list[
            EvidenceClaim
        ] = []

        evidence_counter = 1

        for extraction in extractions:
            for claim in extraction.evidence:
                entity_ids: list[str] = []

                for name in claim.entity_names:
                    try:
                        entity_id = resolve_id(
                            name,
                            extraction.document_id,
                        )
                    except KeyError:
                        continue

                    if entity_id not in entity_ids:
                        entity_ids.append(entity_id)

                canonical_evidence.append(
                    EvidenceClaim(
                        evidence_id=(
                            f"EV_{evidence_counter:03d}"
                        ),
                        document_id=(
                            extraction.document_id
                        ),
                        claim=claim.claim,
                        evidence_type=(
                            claim.evidence_type
                        ),
                        status=claim.status,
                        source_speaker=(
                            claim.source_speaker
                        ),
                        source_excerpt=(
                            claim.source_excerpt
                        ),
                        is_grounded=True,
                        entity_ids=entity_ids,
                        entity_names=(
                            claim.entity_names
                        ),
                        supports=claim.supports,
                        contradicts=(
                            claim.contradicts
                        ),
                    )
                )

                evidence_counter += 1

        # -------------------------------------------------------------
        # Stable relationship IDs.
        # -------------------------------------------------------------
        canonical_relationships: list[
            EntityRelationship
        ] = []

        relationship_counter = 1

        for extraction in extractions:
            for relation in extraction.relationships:
                source_id = resolve_id(
                    relation.source_name,
                    extraction.document_id,
                )

                target_id = resolve_id(
                    relation.target_name,
                    extraction.document_id,
                )

                canonical_relationships.append(
                    EntityRelationship(
                        relationship_id=(
                            f"REL_{relationship_counter:03d}"
                        ),
                        source_id=source_id,
                        source_name=(
                            relation.source_name
                        ),
                        target_id=target_id,
                        target_name=(
                            relation.target_name
                        ),
                        relation=relation.relation,
                        status=relation.status,
                        source_speaker=(
                            relation.source_speaker
                        ),
                        source_excerpt=(
                            relation.source_excerpt
                        ),
                        is_grounded=True,
                        document_id=(
                            extraction.document_id
                        ),
                    )
                )

                relationship_counter += 1

        corpus = CaseEvidenceCorpus(
            case_id="norwood_builder",
            entities=canonical_entities,
            evidence=canonical_evidence,
            relationships=canonical_relationships,
        )

        merged_clusters = [
            {
                "entity_id": entity.entity_id,
                "canonical_name": (
                    entity.canonical_name
                ),
                "aliases": entity.aliases,
                "document_ids": (
                    entity.document_ids
                ),
            }
            for entity in canonical_entities
            if entity.aliases
        ]

        audit = {
            "raw_entity_mentions": sum(
                len(item.entities)
                for item in extractions
            ),
            "canonical_entities": len(
                canonical_entities
            ),
            "evidence_claims": len(
                canonical_evidence
            ),
            "relationships": len(
                canonical_relationships
            ),
            "additional_entity_references_inferred": (
                inferred_mentions
            ),
            "rejected_generic_aliases": (
                len(rejected_aliases)
            ),
            "rejected_alias_examples": (
                rejected_aliases[:20]
            ),
            "merged_entity_clusters": len(
                merged_clusters
            ),
            "largest_alias_clusters": sorted(
                merged_clusters,
                key=lambda item: (
                    -len(item["aliases"]),
                    item["canonical_name"].lower(),
                ),
            )[:15],
            "entity_type_counts": dict(
                Counter(
                    entity.entity_type
                    for entity in canonical_entities
                )
            ),
        }

        return corpus, audit


def save_canonical_outputs(
    corpus: CaseEvidenceCorpus,
    audit: dict,
) -> None:
    CANONICAL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CANONICAL_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            corpus.model_dump(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    with CANONICAL_AUDIT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            audit,
            file,
            indent=2,
            ensure_ascii=False,
        )
