import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.agents.schemas import RetrievedEvidence
from backend.models import CaseEvidenceCorpus, CorpusDocument


CANONICAL_CASE_PATH = Path("data/processed/canonical_case.json")
DOCUMENTS_PATH = Path("data/processed/documents.json")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class AgentEvidenceStore:
    """
    Evidence-level hybrid RAG index.

    - lexical arm: BM25
    - semantic arm: all-MiniLM-L6-v2 cosine similarity
    - fusion: Reciprocal Rank Fusion (RRF)

    The agent layer retrieves canonical EvidenceClaim records rather than
    unconstrained text chunks, so every returned item already carries a
    stable evidence ID, document ID, epistemic status, and verbatim excerpt.
    """

    def __init__(
        self,
        canonical_path: Path = CANONICAL_CASE_PATH,
        documents_path: Path = DOCUMENTS_PATH,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_k: int = 60,
    ):
        self.rrf_k = rrf_k

        with canonical_path.open("r", encoding="utf-8") as file:
            self.case = CaseEvidenceCorpus.model_validate(json.load(file))

        with documents_path.open("r", encoding="utf-8") as file:
            raw_docs = json.load(file)

        self.documents = {
            item["document_id"]: CorpusDocument.model_validate(item)
            for item in raw_docs
        }

        self.evidence = self.case.evidence
        self.evidence_by_id = {
            item.evidence_id: item
            for item in self.evidence
        }

        self.search_texts = [
            " ".join(
                [
                    item.claim,
                    item.source_excerpt,
                    " ".join(item.entity_names),
                    " ".join(item.supports),
                    " ".join(item.contradicts),
                ]
            )
            for item in self.evidence
        ]

        self.bm25 = BM25Okapi(
            [tokenize(text) for text in self.search_texts]
        )

        self.encoder = SentenceTransformer(embedding_model)

        embeddings = self.encoder.encode(
            self.search_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def _bm25_rank(
        self,
        query: str,
        top_k: int,
    ) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(scores)[::-1][:top_k])

    def _semantic_rank(
        self,
        query: str,
        top_k: int,
    ) -> list[int]:
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        return list(np.argsort(scores)[::-1][:top_k])

    def _to_result(
        self,
        index: int,
        score: float,
        source: str,
    ) -> RetrievedEvidence:
        item = self.evidence[index]

        return RetrievedEvidence(
            evidence_id=item.evidence_id,
            document_id=item.document_id,
            claim=item.claim,
            status=item.status,
            evidence_type=item.evidence_type,
            source_excerpt=item.source_excerpt,
            score=float(score),
            source=source,
        )

    def search(
        self,
        query: str,
        top_k: int = 8,
        candidate_pool: int = 20,
    ) -> list[RetrievedEvidence]:
        pool = min(candidate_pool, len(self.evidence))

        bm25_rank = self._bm25_rank(query, pool)
        semantic_rank = self._semantic_rank(query, pool)

        fused: dict[int, float] = {}

        for rank, index in enumerate(bm25_rank, start=1):
            fused[index] = fused.get(index, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        for rank, index in enumerate(semantic_rank, start=1):
            fused[index] = fused.get(index, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        ranked = sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        return [
            self._to_result(index, score, "hybrid_rrf")
            for index, score in ranked
        ]

    def search_many(
        self,
        queries: list[str],
        top_k: int = 10,
        per_query_k: int = 12,
    ) -> list[RetrievedEvidence]:
        """
        Multi-query adversarial retrieval.

        Each query gets an independent hybrid ranking. Their result ranks are
        then fused again with RRF. This is useful for the Fact-Checker because
        "contradictions", "timeline conflicts", and "alternative explanations"
        are distinct retrieval intents.
        """
        fused: dict[str, float] = {}

        for query in queries:
            results = self.search(
                query,
                top_k=per_query_k,
                candidate_pool=max(20, per_query_k),
            )

            for rank, result in enumerate(results, start=1):
                fused[result.evidence_id] = (
                    fused.get(result.evidence_id, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

        ranked_ids = sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        index_by_id = {
            item.evidence_id: index
            for index, item in enumerate(self.evidence)
        }

        return [
            self._to_result(
                index_by_id[evidence_id],
                score,
                "multi_query_rrf",
            )
            for evidence_id, score in ranked_ids
        ]

    def get_by_ids(
        self,
        evidence_ids: list[str],
    ) -> tuple[list[RetrievedEvidence], list[str]]:
        results: list[RetrievedEvidence] = []
        missing: list[str] = []

        for evidence_id in evidence_ids:
            item = self.evidence_by_id.get(evidence_id)

            if item is None:
                missing.append(evidence_id)
                continue

            results.append(
                RetrievedEvidence(
                    evidence_id=item.evidence_id,
                    document_id=item.document_id,
                    claim=item.claim,
                    status=item.status,
                    evidence_type=item.evidence_type,
                    source_excerpt=item.source_excerpt,
                    score=1.0,
                    source="explicit_citation",
                )
            )

        return results, missing
