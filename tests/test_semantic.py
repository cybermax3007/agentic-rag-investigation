from backend.retrieval.semantic_retriever import (
    SemanticRetriever
)


def main():

    print(
        "Loading semantic retriever..."
    )

    retriever = SemanticRetriever()

    print(
        f"Loaded "
        f"{len(retriever.documents)} "
        f"documents."
    )

    print()

    query = (
        "thumbprint McFarlane"
    )

    print(
        f"Query: {query}"
    )

    print(
        "-" * 60
    )

    results = retriever.search(
        query=query,
        top_k=5
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"RANK {rank}"
        )

        print(
            f"Document: "
            f"{result.document_id}"
        )

        print(
            f"Similarity: "
            f"{result.score:.4f}"
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

        print(
            preview
        )

        print()
        print(
            "-" * 60
        )


if __name__ == "__main__":
    main()