from backend.agents.evidence_store import AgentEvidenceStore


BENCHMARK = [
    {
        "query": "blood thumbprint McFarlane",
        "expected_document": "DOC_012",
    },
    {
        "query": "blood-stained walking stick McFarlane",
        "expected_document": "DOC_003",
    },
    {
        "query": "Lestrade telegram fresh evidence guilt established",
        "expected_document": "DOC_011",
    },
    {
        "query": "Oldacre revenge frame McFarlane disappearance",
        "expected_document": "DOC_016",
    },
    {
        "query": "fresh blood stains bedroom",
        "expected_document": "DOC_010",
    },
]


def main() -> None:
    store = AgentEvidenceStore()

    hits = 0

    print("=" * 72)
    print("CASEFILE AI — RETRIEVAL EVALUATION")
    print("=" * 72)

    for item in BENCHMARK:
        results = store.search(
            item["query"],
            top_k=5,
        )

        docs = [
            result.document_id
            for result in results
        ]

        hit = (
            item["expected_document"]
            in docs
        )

        hits += int(hit)

        print(
            f"{'HIT ' if hit else 'MISS'} | "
            f"{item['query']}"
        )
        print(
            f"       expected: {item['expected_document']} | "
            f"top-5 docs: {docs}"
        )

    hit_at_5 = hits / len(BENCHMARK)

    print("-" * 72)
    print(
        f"Hit@5: {hits}/{len(BENCHMARK)} "
        f"= {hit_at_5:.1%}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
