import argparse
import json
import time
from collections import Counter
from pathlib import Path

from backend.extraction.evidence_extractor import (
    EvidenceExtractor,
    validate_excerpt,
)
from backend.models import CorpusDocument, DocumentExtraction


EXTRACTIONS_DIR = Path("data/processed/extractions")
RAW_EXTRACTIONS_FILE = Path("data/processed/raw_extractions.json")
ERRORS_FILE = Path("data/processed/extraction_errors.json")
REJECTED_FILE = Path("data/processed/rejected_extraction_items.json")
AUDIT_FILE = Path("data/processed/extraction_audit.json")
CORPUS_PATH = Path("data/processed/documents.json")


def load_corpus() -> list[CorpusDocument]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {CORPUS_PATH}"
        )

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        raw_documents = json.load(file)

    corpus = [
        CorpusDocument(**document)
        for document in raw_documents
    ]

    corpus.sort(key=lambda document: document.document_id)
    return corpus


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evidence extraction for CaseFile AI"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-extract every document.",
    )

    args = parser.parse_args()

    EXTRACTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus = load_corpus()
    extractor = EvidenceExtractor()

    extractions_by_id: dict[
        str,
        DocumentExtraction,
    ] = {}

    errors: list[dict] = []
    rejected_items: list[dict] = []

    newly_extracted_count = 0
    cached_count = 0

    print("=" * 76)
    print("CASEFILE AI — BATCH EVIDENCE EXTRACTION")
    print(
        f"Corpus: {len(corpus)} documents "
        f"({corpus[0].document_id} -> "
        f"{corpus[-1].document_id})"
    )
    print(
        f"Model: {extractor.model_name}"
    )
    print(
        f"Force re-extraction: {args.force}"
    )
    print("=" * 76)

    for index, document in enumerate(
        corpus,
        start=1,
    ):
        cache_path = (
            EXTRACTIONS_DIR
            / f"{document.document_id}.json"
        )

        if (
            not args.force
            and cache_path.exists()
        ):
            try:
                with cache_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    cached_data = json.load(file)

                extraction = (
                    DocumentExtraction.model_validate(
                        cached_data
                    )
                )

                extraction, rejected = (
                    extractor.validate_extraction(
                        document,
                        extraction,
                    )
                )

                rejected_items.extend(rejected)

                extractions_by_id[
                    document.document_id
                ] = extraction

                cached_count += 1

                print(
                    f"[{index:02d}/{len(corpus):02d}] "
                    f"{document.document_id}: cache "
                    f"({len(extraction.evidence)} claims, "
                    f"{len(extraction.relationships)} relations)"
                )

                continue

            except Exception as exc:
                print(
                    f"[{index:02d}/{len(corpus):02d}] "
                    f"{document.document_id}: invalid cache "
                    f"({exc}); re-extracting"
                )

        print(
            f"[{index:02d}/{len(corpus):02d}] "
            f"{document.document_id}: extracting..."
        )

        started = time.time()

        try:
            extraction = extractor.extract_document(
                document
            )

            extraction, rejected = (
                extractor.validate_extraction(
                    document,
                    extraction,
                )
            )

            rejected_items.extend(rejected)

            # Persist each completed document immediately.
            save_json(
                cache_path,
                extraction.model_dump(),
            )

            extractions_by_id[
                document.document_id
            ] = extraction

            newly_extracted_count += 1
            elapsed = time.time() - started

            print(
                f"    -> saved in {elapsed:.1f}s | "
                f"{len(extraction.entities)} entities | "
                f"{len(extraction.evidence)} claims | "
                f"{len(extraction.relationships)} relations | "
                f"{len(rejected)} rejected"
            )

        except Exception as exc:
            print(
                f"    [ERROR] {document.document_id}: "
                f"{exc}"
            )

            errors.append(
                {
                    "document_id": (
                        document.document_id
                    ),
                    "error": str(exc),
                }
            )

    aggregated = [
        extractions_by_id[
            document.document_id
        ].model_dump()
        for document in corpus
        if document.document_id
        in extractions_by_id
    ]

    save_json(
        RAW_EXTRACTIONS_FILE,
        aggregated,
    )

    save_json(
        REJECTED_FILE,
        rejected_items,
    )

    if errors:
        save_json(
            ERRORS_FILE,
            errors,
        )
    elif ERRORS_FILE.exists():
        ERRORS_FILE.unlink()

    # ================================================================
    # AUDIT
    # ================================================================

    document_text = {
        document.document_id: document.text
        for document in corpus
    }

    total_entities = 0
    total_claims = 0
    total_relationships = 0

    matched_claim_excerpts = 0
    matched_relationship_excerpts = 0

    claim_status_counts = Counter()
    relationship_status_counts = Counter()
    evidence_type_counts = Counter()
    entity_name_counts = Counter()

    for (
        document_id,
        extraction,
    ) in extractions_by_id.items():
        source_text = document_text[
            document_id
        ]

        total_entities += len(
            extraction.entities
        )

        for entity in extraction.entities:
            entity_name_counts[
                entity.name
            ] += 1

        for claim in extraction.evidence:
            total_claims += 1

            claim_status_counts[
                claim.status
            ] += 1

            evidence_type_counts[
                claim.evidence_type
            ] += 1

            if validate_excerpt(
                source_text,
                claim.source_excerpt,
            ):
                matched_claim_excerpts += 1

        for relation in (
            extraction.relationships
        ):
            total_relationships += 1

            relationship_status_counts[
                relation.status
            ] += 1

            if validate_excerpt(
                source_text,
                relation.source_excerpt,
            ):
                matched_relationship_excerpts += 1

    rejected_claims = sum(
        item.get("kind") == "claim"
        for item in rejected_items
    )

    rejected_relationships = sum(
        item.get("kind")
        == "relationship"
        for item in rejected_items
    )

    audit = {
        "documents_total": len(corpus),
        "documents_processed": (
            len(extractions_by_id)
        ),
        "documents_failed": len(errors),
        "documents_newly_extracted": (
            newly_extracted_count
        ),
        "documents_loaded_from_cache": (
            cached_count
        ),
        "total_raw_entities": total_entities,
        "total_claims_after_validation": (
            total_claims
        ),
        "total_relationships_after_validation": (
            total_relationships
        ),
        "claim_excerpt_matches": (
            matched_claim_excerpts
        ),
        "relationship_excerpt_matches": (
            matched_relationship_excerpts
        ),
        "rejected_claims": rejected_claims,
        "rejected_relationships": (
            rejected_relationships
        ),
        "claim_status_counts": dict(
            claim_status_counts
        ),
        "relationship_status_counts": dict(
            relationship_status_counts
        ),
        "evidence_type_counts": dict(
            evidence_type_counts
        ),
        "top_entity_name_variants": (
            entity_name_counts.most_common(15)
        ),
    }

    save_json(
        AUDIT_FILE,
        audit,
    )

    claim_match_rate = (
        matched_claim_excerpts
        / total_claims
        * 100
        if total_claims
        else 0.0
    )

    relation_match_rate = (
        matched_relationship_excerpts
        / total_relationships
        * 100
        if total_relationships
        else 0.0
    )

    print()
    print("=" * 76)
    print("CORPUS EXTRACTION AUDIT")
    print("=" * 76)
    print(
        f"Documents processed          : "
        f"{len(extractions_by_id)} / "
        f"{len(corpus)}"
    )
    print(
        f"Failed documents             : "
        f"{len(errors)}"
    )
    print(
        f"New API extractions          : "
        f"{newly_extracted_count}"
    )
    print(
        f"Loaded from cache            : "
        f"{cached_count}"
    )
    print(
        f"Raw entity mentions          : "
        f"{total_entities}"
    )
    print(
        f"Validated claims             : "
        f"{total_claims}"
    )
    print(
        f"Validated relationships      : "
        f"{total_relationships}"
    )
    print(
        f"Rejected claim excerpts      : "
        f"{rejected_claims}"
    )
    print(
        f"Rejected relationship excerpts: "
        f"{rejected_relationships}"
    )
    print(
        f"Claim excerpt match rate     : "
        f"{claim_match_rate:.2f}%"
    )
    print(
        f"Relation excerpt match rate  : "
        f"{relation_match_rate:.2f}%"
    )

    print("-" * 76)
    print("Claim status counts")
    for status, count in sorted(
        claim_status_counts.items()
    ):
        print(
            f"  {status:<12} {count}"
        )

    print("-" * 76)
    print("Relationship status counts")
    for status, count in sorted(
        relationship_status_counts.items()
    ):
        print(
            f"  {status:<12} {count}"
        )

    print("-" * 76)
    print("Evidence type counts")
    for evidence_type, count in sorted(
        evidence_type_counts.items()
    ):
        print(
            f"  {evidence_type:<12} {count}"
        )

    print("-" * 76)
    print("Top repeated entity-name variants")
    for name, count in (
        entity_name_counts.most_common(15)
    ):
        print(
            f"  {name:<35} {count}"
        )

    print("=" * 76)
    print(
        f"Aggregated output: "
        f"{RAW_EXTRACTIONS_FILE}"
    )
    print(
        f"Audit output     : "
        f"{AUDIT_FILE}"
    )
    print(
        f"Rejected items   : "
        f"{REJECTED_FILE}"
    )


if __name__ == "__main__":
    main()
