import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.models import CorpusDocument, SearchResult


DEFAULT_CORPUS_PATH = Path(
    "data/processed/documents.json"
)

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


class SemanticRetriever:

    def __init__(
        self,
        corpus_path: Path = DEFAULT_CORPUS_PATH
    ):

        self.corpus_path = corpus_path

        self.documents = (
            self._load_documents()
        )

        print(
            f"Loading semantic model: "
            f"{MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Creating document embeddings..."
        )

        document_texts = [
            document.text
            for document in self.documents
        ]

        self.document_embeddings = (
            self.model.encode(
                document_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        )

    def _load_documents(
        self
    ) -> list[CorpusDocument]:

        if not self.corpus_path.exists():

            raise FileNotFoundError(
                f"Corpus not found: "
                f"{self.corpus_path}"
            )

        with self.corpus_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            raw_documents = json.load(file)

        documents = [
            CorpusDocument(**document)
            for document in raw_documents
        ]

        return documents

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[SearchResult]:

        if not query.strip():
            return []

        query_embedding = (
            self.model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]
        )

        scores = np.dot(
            self.document_embeddings,
            query_embedding
        )

        ranked_indices = (
            np.argsort(scores)[::-1]
        )

        results = []

        for index in ranked_indices[:top_k]:

            document = self.documents[
                int(index)
            ]

            result = SearchResult(
                document_id=(
                    document.document_id
                ),
                score=float(scores[index]),
                text=document.text,
                source_title=(
                    document.source_title
                ),
                paragraph_start=(
                    document.paragraph_start
                ),
                paragraph_end=(
                    document.paragraph_end
                )
            )

            results.append(result)

        return results