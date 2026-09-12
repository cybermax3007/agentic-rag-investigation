from pydantic import BaseModel


class CorpusDocument(BaseModel):
    document_id: str
    case_id: str
    source_title: str
    source_url: str
    paragraph_start: int
    paragraph_end: int
    word_count: int
    text: str


class SearchResult(BaseModel):
    document_id: str
    score: float
    text: str
    source_title: str
    paragraph_start: int
    paragraph_end: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]