from backend.retrieval.hybrid_retriever import (
    HybridRetriever
)


def main():

    print("Loading hybrid retriever...")

    retriever = HybridRetriever()

    print()
    print("Hybrid retriever ready.")
    print()

    query = "thumbprint McFarlane"

    print(f"Query: {query}")
    print("-" * 60)

    results = retriever.search(
        query=query,
        top_k=5
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(f"RANK {rank}")

        print(
            f"Document: "
            f"{result.document_id}"
        )

        print(
            f"RRF score: "
            f"{result.score:.6f}"
        )

        print(
            f"Paragraphs: "
            f"{result.paragraph_start}"
            f"-"
            f"{result.paragraph_end}"
        )

        print()

        preview = (
            result.text[:500]
            .replace("\n", " ")
        )

        print(preview)

        print()
        print("-" * 60)

    retriever.compare_rankings(
        query=query,
        top_k=5
    )


if __name__ == "__main__":
    main()