from backend.extraction.canonicalizer import (
    CANONICAL_AUDIT_PATH,
    CANONICAL_OUTPUT_PATH,
    EvidenceCanonicalizer,
    save_canonical_outputs,
)


def main() -> None:
    print("=" * 72)
    print("CASEFILE AI — ENTITY CANONICALIZATION")
    print("=" * 72)

    canonicalizer = EvidenceCanonicalizer()

    corpus, audit = canonicalizer.canonicalize()

    save_canonical_outputs(
        corpus,
        audit,
    )

    print(
        f"Raw entity mentions             : "
        f"{audit['raw_entity_mentions']}"
    )
    print(
        f"Canonical entities              : "
        f"{audit['canonical_entities']}"
    )
    print(
        f"Evidence claims                 : "
        f"{audit['evidence_claims']}"
    )
    print(
        f"Canonical relationships         : "
        f"{audit['relationships']}"
    )
    print(
        f"Inferred missing entity refs    : "
        f"{audit['additional_entity_references_inferred']}"
    )
    print(
        f"Merged entity clusters          : "
        f"{audit['merged_entity_clusters']}"
    )

    print("-" * 72)
    print("Entity type counts")

    for entity_type, count in sorted(
        audit["entity_type_counts"].items()
    ):
        print(
            f"  {entity_type:<14} {count}"
        )

    print("-" * 72)
    print("Largest alias clusters")

    for cluster in audit[
        "largest_alias_clusters"
    ]:
        print(
            f"  {cluster['entity_id']} | "
            f"{cluster['canonical_name']}"
        )
        print(
            f"      aliases: "
            f"{cluster['aliases']}"
        )
        print(
            f"      docs: "
            f"{cluster['document_ids']}"
        )

    print("=" * 72)
    print(
        f"Canonical corpus: "
        f"{CANONICAL_OUTPUT_PATH}"
    )
    print(
        f"Audit           : "
        f"{CANONICAL_AUDIT_PATH}"
    )


if __name__ == "__main__":
    main()
