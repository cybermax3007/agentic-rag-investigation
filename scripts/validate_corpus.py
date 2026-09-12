import json
from pathlib import Path


DOCUMENTS_PATH = Path(
    "data/processed/documents.json"
)


REQUIRED_FIELDS = {
    "document_id",
    "case_id",
    "source_title",
    "source_url",
    "paragraph_start",
    "paragraph_end",
    "word_count",
    "text",
}


def main():

    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            "documents.json does not exist."
        )

    with DOCUMENTS_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError(
            "Corpus must be a JSON list."
        )

    if len(documents) == 0:
        raise ValueError(
            "Corpus contains no documents."
        )

    seen_ids = set()

    for document in documents:

        missing = (
            REQUIRED_FIELDS
            - set(document.keys())
        )

        if missing:
            raise ValueError(
                f"{document.get('document_id')} "
                f"is missing fields: {missing}"
            )

        document_id = document[
            "document_id"
        ]

        if document_id in seen_ids:
            raise ValueError(
                f"Duplicate document ID: "
                f"{document_id}"
            )

        seen_ids.add(
            document_id
        )

        if not document["text"].strip():
            raise ValueError(
                f"{document_id} has empty text."
            )

        if document["word_count"] <= 0:
            raise ValueError(
                f"{document_id} has invalid "
                f"word count."
            )

    print(
        "Corpus validation passed."
    )

    print(
        f"Documents checked: "
        f"{len(documents)}"
    )

    print(
        "All document IDs are unique."
    )

    print(
        "All required fields are present."
    )

    print(
        "No empty documents found."
    )


if __name__ == "__main__":
    main()