import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.models import CorpusDocument, SearchResult


DEFAULT_CORPUS_PATH = Path(
    "data/processed/documents.json"
)


def tokenize(text: str) -> list[str]:
    

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


class BM25Retriever:

    def __init__(
        self,
        corpus_path: Path = DEFAULT_CORPUS_PATH
    ):

        self.corpus_path = corpus_path

        self.documents = (
            self._load_documents()
        )

        self.tokenized_corpus = [
            tokenize(document.text)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
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

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices[:top_k]:

            document = self.documents[index]

            result = SearchResult(
                document_id=document.document_id,
                score=float(scores[index]),
                text=document.text,
                source_title=document.source_title,
                paragraph_start=document.paragraph_start,
                paragraph_end=document.paragraph_end
            )

            results.append(result)

        return results