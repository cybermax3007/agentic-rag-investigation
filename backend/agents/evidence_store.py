import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.agents.schemas import RetrievedEvidence
from backend.models import CaseEvidenceCorpus, CorpusDocument


CANONICAL_CASE_PATH = Path("data/processed/canonical_case.json")
DOCUMENTS_PATH = Path("data/processed/documents.json")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_phrase(text: str) -> str:
    return " ".join(tokenize(text))


class AgentEvidenceStore:
    """
    Evidence-level retrieval for the agent layer.

    Base retrieval:
      - lexical BM25
      - semantic MiniLM embeddings
      - Reciprocal Rank Fusion (RRF)

    Bonus retrieval signals:
      - graph-aware entity RRF arm when the query names a canonical entity
      - optional cross-encoder reranking over the fused candidate pool

    Every result already carries a stable EV_### ID, source DOC_### ID,
    epistemic status, and verbatim grounding excerpt.
    """

    def __init__(
        self,
        canonical_path: Path = CANONICAL_CASE_PATH,
        documents_path: Path = DOCUMENTS_PATH,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rrf_k: int = 60,
        graph_weight: float = 0.75,
        enable_reranker: bool | None = None,
    ):
        self.rrf_k = rrf_k
        self.graph_weight = graph_weight

        if enable_reranker is None:
            enable_reranker = (
                os.getenv("ENABLE_RERANKER", "true").lower()
                in {"1", "true", "yes", "on"}
            )

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
        self.index_by_evidence_id = {
            item.evidence_id: index
            for index, item in enumerate(self.evidence)
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

        self.entity_phrases: list[tuple[str, str]] = []
        for entity in self.case.entities:
            phrases = {
                entity.canonical_name,
                *entity.aliases,
            }
            for phrase in phrases:
                normalized = normalize_phrase(phrase)
                if len(normalized) >= 3:
                    self.entity_phrases.append(
                        (normalized, entity.entity_id)
                    )

        self.reranker = None
        if enable_reranker:
            try:
                self.reranker = CrossEncoder(reranker_model)
            except Exception as exc:
                # Retrieval remains fully functional without the bonus reranker.
                print(
                    "[retrieval] Cross-encoder unavailable; "
                    f"falling back to RRF only: {exc}"
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

    def _query_entity_ids(self, query: str) -> set[str]:
        normalized_query = f" {normalize_phrase(query)} "
        matched: set[str] = set()

        # Longest names first avoids a short alias dominating the match logic.
        for phrase, entity_id in sorted(
            self.entity_phrases,
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if f" {phrase} " in normalized_query:
                matched.add(entity_id)

        return matched

    def _should_use_graph(self, query: str) -> bool:
        """
        Use the graph arm only when the query is genuinely relational.

        A person name by itself is not enough: broad entity-neighborhood
        evidence can dilute precise factual retrieval. Two named entities, or
        one named entity plus an explicit relationship cue, is a stronger
        signal that graph expansion is useful.
        """
        entity_ids = self._query_entity_ids(query)

        if len(entity_ids) >= 2:
            return True

        if not entity_ids:
            return False

        relation_cues = {
            "relationship",
            "relation",
            "connect",
            "connected",
            "connection",
            "link",
            "linked",
            "between",
            "associate",
            "associated",
            "know",
            "knew",
            "met",
            "visited",
            "visit",
            "mother",
            "son",
            "client",
            "lawyer",
            "witness",
            "owner",
            "employer",
            "employee",
        }

        tokens = set(tokenize(query))
        return bool(tokens.intersection(relation_cues))

    def _graph_rank(
        self,
        query: str,
        top_k: int,
    ) -> list[int]:
        entity_ids = self._query_entity_ids(query)
        if not entity_ids:
            return []

        rows: list[tuple[int, int, int]] = []

        for index, evidence in enumerate(self.evidence):
            overlap = len(
                entity_ids.intersection(evidence.entity_ids)
            )
            if overlap == 0:
                continue

            # Within the graph arm only, prefer direct overlap and then
            # verified evidence as a deterministic tie-break.
            verified_bonus = int(evidence.status == "verified")
            rows.append((index, overlap, verified_bonus))

        rows.sort(
            key=lambda row: (
                -row[1],
                -row[2],
                self.evidence[row[0]].evidence_id,
            )
        )

        return [row[0] for row in rows[:top_k]]

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

    def search_lexical(
        self,
        query: str,
        top_k: int = 8,
    ) -> list[RetrievedEvidence]:
        scores = self.bm25.get_scores(tokenize(query))
        indices = list(np.argsort(scores)[::-1][:top_k])

        return [
            self._to_result(
                index,
                float(scores[index]),
                "bm25",
            )
            for index in indices
        ]

    def search_semantic(
        self,
        query: str,
        top_k: int = 8,
    ) -> list[RetrievedEvidence]:
        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self.embeddings @ np.asarray(
            query_embedding,
            dtype=np.float32,
        )
        indices = list(np.argsort(scores)[::-1][:top_k])

        return [
            self._to_result(
                index,
                float(scores[index]),
                "semantic",
            )
            for index in indices
        ]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 8,
        candidate_pool: int = 20,
        include_graph: bool = False,
    ) -> list[RetrievedEvidence]:
        pool = min(candidate_pool, len(self.evidence))

        # The retrieval arms are independent. Run them concurrently so the
        # hybrid layer performs genuine parallel retrieval rather than
        # sequentially waiting for lexical, dense, and graph signals.
        max_workers = 3 if include_graph else 2

        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            bm25_future = executor.submit(
                self._bm25_rank,
                query,
                pool,
            )
            semantic_future = executor.submit(
                self._semantic_rank,
                query,
                pool,
            )
            graph_future = (
                executor.submit(
                    self._graph_rank,
                    query,
                    pool,
                )
                if include_graph
                else None
            )

            bm25_rank = bm25_future.result()
            semantic_rank = semantic_future.result()
            graph_rank = (
                graph_future.result()
                if graph_future is not None
                else []
            )

        fused: dict[int, float] = {}

        for rank, index in enumerate(bm25_rank, start=1):
            fused[index] = fused.get(index, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        for rank, index in enumerate(semantic_rank, start=1):
            fused[index] = fused.get(index, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        for rank, index in enumerate(graph_rank, start=1):
            fused[index] = fused.get(index, 0.0) + (
                self.graph_weight
                / (self.rrf_k + rank)
            )

        ranked = sorted(
            fused.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        source = (
            "parallel_bm25+semantic+graph_rrf"
            if graph_rank
            else "parallel_bm25+semantic_rrf"
        )

        return [
            self._to_result(index, score, source)
            for index, score in ranked
        ]

    def search(
        self,
        query: str,
        top_k: int = 8,
        candidate_pool: int = 24,
        rerank_pool: int = 14,
    ) -> list[RetrievedEvidence]:
        """
        Production retrieval path.

        1. BM25 + semantic + optional graph-aware RRF.
        2. Cross-encoder reranking of the best fused candidates when the
           reranker is available.
        """
        fused = self.search_hybrid(
            query=query,
            top_k=max(top_k, rerank_pool),
            candidate_pool=candidate_pool,
            include_graph=self._should_use_graph(query),
        )

        if self.reranker is None:
            return fused[:top_k]

        pairs = [
            [
                query,
                self.search_texts[
                    self.index_by_evidence_id[item.evidence_id]
                ],
            ]
            for item in fused
        ]

        rerank_scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(fused, rerank_scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )[:top_k]

        results: list[RetrievedEvidence] = []

        for item, score in reranked:
            item.score = float(score)
            item.source = (
                item.source + "+cross_encoder"
            )
            results.append(item)

        return results

    def search_many(
        self,
        queries: list[str],
        top_k: int = 10,
        per_query_k: int = 12,
    ) -> list[RetrievedEvidence]:
        """
        Multi-query adversarial retrieval used by the Fact-Checker.
        Each intent receives an independent production retrieval, then
        result ranks are fused with another RRF pass.
        """
        fused: dict[str, float] = {}

        for query in queries:
            results = self.search(
                query,
                top_k=per_query_k,
                candidate_pool=max(24, per_query_k * 2),
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

        return [
            self._to_result(
                self.index_by_evidence_id[evidence_id],
                score,
                "multi_query_rrf",
            )
            for evidence_id, score in ranked_ids
        ]

    def verified_evidence(
        self,
    ) -> list[RetrievedEvidence]:
        """
        Return every verified evidence claim in the case.

        The corpus is deliberately small (87 claims), so before an agent
        declares a material fact unresolved we can afford a deterministic
        safety pass over the complete verified subset. This is not used as
        the primary retriever; it is a bounded completeness guard.
        """
        results: list[RetrievedEvidence] = []

        for index, item in enumerate(self.evidence):
            if item.status != "verified":
                continue

            results.append(
                self._to_result(
                    index=index,
                    score=1.0,
                    source="verified_safety_pass",
                )
            )

        return results

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
