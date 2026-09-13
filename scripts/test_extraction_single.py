import argparse
import json
from pathlib import Path

from backend.extraction.evidence_extractor import EvidenceExtractor
from backend.models import CorpusDocument


CORPUS_PATH = Path("data/processed/documents.json")


def load_document(document_id: str) -> CorpusDocument:
    with CORPUS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_documents = json.load(file)

    for raw_document in raw_documents:
        if raw_document["document_id"] == document_id:
            return CorpusDocument(**raw_document)

    raise ValueError(
        f"Document not found: {document_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc",
        default="DOC_016",
        help="Document ID to extract.",
    )
    args = parser.parse_args()

    document = load_document(args.doc)
    extractor = EvidenceExtractor()

    print(
        f"Extracting {document.document_id}..."
    )

    extraction = extractor.extract_document(
        document
    )

    extraction, rejected = (
        extractor.validate_extraction(
            document,
            extraction,
        )
    )

    print()
    print("=" * 70)
    print("ENTITIES")
    print("=" * 70)

    for entity in extraction.entities:
        print(
            f"{entity.entity_type:<13} "
            f"{entity.name}"
        )

    print()
    print("=" * 70)
    print("CLAIMS")
    print("=" * 70)

    for index, claim in enumerate(
        extraction.evidence,
        start=1,
    ):
        print(
            f"[{index}] "
            f"{claim.evidence_type.upper()} | "
            f"{claim.status.upper()}"
        )
        print(
            f"Claim: {claim.claim}"
        )
        print(
            f"Excerpt: {claim.source_excerpt}"
        )

        if claim.source_speaker:
            print(
                f"Speaker: "
                f"{claim.source_speaker}"
            )

        if claim.supports:
            print(
                f"Supports: {claim.supports}"
            )

        if claim.contradicts:
            print(
                f"Contradicts: "
                f"{claim.contradicts}"
            )

        print()

    print("=" * 70)
    print("RELATIONSHIPS")
    print("=" * 70)

    for relation in (
        extraction.relationships
    ):
        print(
            f"{relation.source_name} "
            f"--[{relation.relation} | "
            f"{relation.status}]--> "
            f"{relation.target_name}"
        )
        print(
            f"Excerpt: "
            f"{relation.source_excerpt}"
        )
        print()

    print("=" * 70)
    print(
        f"Rejected ungrounded items: "
        f"{len(rejected)}"
    )

    for item in rejected:
        print(item)


if __name__ == "__main__":
    main()
