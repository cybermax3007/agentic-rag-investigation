from backend.models import SearchResult
from backend.retrieval.bm25_retriever import BM25Retriever
from backend.retrieval.semantic_retriever import SemanticRetriever


RRF_K = 60


class HybridRetriever:

    def __init__(self):

        print("Loading BM25 retriever...")
        self.bm25 = BM25Retriever()

        print("Loading semantic retriever...")
        self.semantic = SemanticRetriever()

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10
    ) -> list[SearchResult]:

        if not query.strip():
            return []

        
        bm25_results = self.bm25.search(
            query=query,
            top_k=candidate_k
        )

        semantic_results = self.semantic.search(
            query=query,
            top_k=candidate_k
        )

        
        rrf_scores = {}

        
        document_lookup = {}

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            document_id = result.document_id

            document_lookup[document_id] = result

            if document_id not in rrf_scores:
                rrf_scores[document_id] = 0.0

            rrf_scores[document_id] += (
                1 / (RRF_K + rank)
            )

        
        for rank, result in enumerate(
            semantic_results,
            start=1
        ):

            document_id = result.document_id

            document_lookup[document_id] = result

            if document_id not in rrf_scores:
                rrf_scores[document_id] = 0.0

            rrf_scores[document_id] += (
                1 / (RRF_K + rank)
            )

        
        ranked_document_ids = sorted(
            rrf_scores.keys(),
            key=lambda document_id: rrf_scores[
                document_id
            ],
            reverse=True
        )

        results = []

        for document_id in ranked_document_ids[:top_k]:

            original_result = document_lookup[
                document_id
            ]

            hybrid_result = SearchResult(
                document_id=document_id,
                score=rrf_scores[document_id],
                text=original_result.text,
                source_title=(
                    original_result.source_title
                ),
                paragraph_start=(
                    original_result.paragraph_start
                ),
                paragraph_end=(
                    original_result.paragraph_end
                )
            )

            results.append(
                hybrid_result
            )

        return results

    def compare_rankings(
        self,
        query: str,
        top_k: int = 5
    ) -> None:

        bm25_results = self.bm25.search(
            query=query,
            top_k=top_k
        )

        semantic_results = self.semantic.search(
            query=query,
            top_k=top_k
        )

        hybrid_results = self.search(
            query=query,
            top_k=top_k,
            candidate_k=max(top_k, 10)
        )

        print()
        print("=" * 60)

        print("BM25 ranking:")
        print("-" * 60)

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):
            print(
                f"{rank}. "
                f"{result.document_id} "
                f"| score = "
                f"{result.score:.4f}"
            )

        print()
        print("Semantic ranking:")
        print("-" * 60)

        for rank, result in enumerate(
            semantic_results,
            start=1
        ):
            print(
                f"{rank}. "
                f"{result.document_id} "
                f"| similarity = "
                f"{result.score:.4f}"
            )

        print()
        print("Hybrid RRF ranking:")
        print("-" * 60)

        for rank, result in enumerate(
            hybrid_results,
            start=1
        ):
            print(
                f"{rank}. "
                f"{result.document_id} "
                f"| RRF = "
                f"{result.score:.6f}"
            )

        print("=" * 60)