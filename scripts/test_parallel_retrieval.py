import time

from backend.agents.evidence_store import AgentEvidenceStore


def main() -> None:
    store = AgentEvidenceStore(
        enable_reranker=False,
    )

    def slow_rank(_query: str, _top_k: int) -> list[int]:
        time.sleep(0.30)
        return [0, 1, 2]

    store._bm25_rank = slow_rank
    store._semantic_rank = slow_rank
    store._graph_rank = slow_rank

    started = time.perf_counter()
    results = store.search_hybrid(
        "relationship between McFarlane and Oldacre",
        top_k=3,
        candidate_pool=3,
        include_graph=True,
    )
    elapsed = time.perf_counter() - started

    # Three sequential 0.30 s retrieval arms would take about 0.90 s.
    # Leave generous headroom for CI and slower machines while still proving
    # that the arms overlap in time.
    assert elapsed < 0.72, (
        f"Retrieval arms appear sequential: {elapsed:.3f}s"
    )
    assert results, "Parallel hybrid retrieval returned no results."
    assert "parallel_" in results[0].source

    print("Parallel retrieval regression test: PASS")
    print(f"Elapsed for three 0.30s arms: {elapsed:.3f}s")
    print(f"Source label: {results[0].source}")


if __name__ == "__main__":
    main()
