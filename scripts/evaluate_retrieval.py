from collections.abc import Callable

from backend.agents.evidence_store import AgentEvidenceStore
from backend.agents.schemas import RetrievedEvidence


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
    {
        "query": "direct observation of Oldacre emerging alive from a concealed room",
        "expected_document": "DOC_014",
    },
]


def reciprocal_rank(
    results: list[RetrievedEvidence],
    expected_document: str,
) -> float:
    for rank, result in enumerate(results, start=1):
        if result.document_id == expected_document:
            return 1.0 / rank
    return 0.0


def evaluate(
    name: str,
    retriever: Callable[[str, int], list[RetrievedEvidence]],
) -> dict:
    hit1 = 0
    hit3 = 0
    hit5 = 0
    rr_total = 0.0

    print("-" * 78)
    print(name)
    print("-" * 78)

    for item in BENCHMARK:
        results = retriever(item["query"], 5)
        docs = [result.document_id for result in results]

        expected = item["expected_document"]

        hit1 += int(expected in docs[:1])
        hit3 += int(expected in docs[:3])
        hit5 += int(expected in docs[:5])
        rr_total += reciprocal_rank(results, expected)

        print(
            f"{'HIT ' if expected in docs else 'MISS'} | "
            f"{item['query']}"
        )
        print(
            f"       expected: {expected} | top-5 docs: {docs}"
        )

    n = len(BENCHMARK)

    metrics = {
        "Hit@1": hit1 / n,
        "Hit@3": hit3 / n,
        "Hit@5": hit5 / n,
        "MRR@5": rr_total / n,
    }

    print(
        "       "
        + " | ".join(
            f"{key}: {value:.1%}"
            for key, value in metrics.items()
        )
    )

    return metrics


def main() -> None:
    store = AgentEvidenceStore()

    print("=" * 78)
    print("CASEFILE AI — RETRIEVAL ABLATION")
    print("=" * 78)

    rows = {}

    rows["BM25"] = evaluate(
        "1) BM25 lexical retrieval",
        lambda query, top_k: store.search_lexical(
            query,
            top_k=top_k,
        ),
    )

    rows["Semantic"] = evaluate(
        "2) MiniLM semantic retrieval",
        lambda query, top_k: store.search_semantic(
            query,
            top_k=top_k,
        ),
    )

    rows["Hybrid RRF"] = evaluate(
        "3) BM25 + semantic RRF",
        lambda query, top_k: store.search_hybrid(
            query,
            top_k=top_k,
            include_graph=False,
        ),
    )

    rows["Graph RRF"] = evaluate(
        "4) BM25 + semantic + graph-aware RRF",
        lambda query, top_k: store.search_hybrid(
            query,
            top_k=top_k,
            include_graph=True,
        ),
    )

    rows["Production"] = evaluate(
        "5) Production retrieval (+ cross-encoder when available)",
        lambda query, top_k: store.search(
            query,
            top_k=top_k,
        ),
    )

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        f"{'Retriever':<18} "
        f"{'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'MRR@5':>8}"
    )

    for name, metrics in rows.items():
        print(
            f"{name:<18} "
            f"{metrics['Hit@1']:>7.1%} "
            f"{metrics['Hit@3']:>7.1%} "
            f"{metrics['Hit@5']:>7.1%} "
            f"{metrics['MRR@5']:>7.1%}"
        )


if __name__ == "__main__":
    main()
