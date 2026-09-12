import json
import re
from pathlib import Path


RAW_PATH = Path("data/raw/norwood_builder.txt")
OUTPUT_PATH = Path("data/processed/documents.json")

CASE_ID = "norwood_builder"

SOURCE_TITLE = "The Adventure of the Norwood Builder"

SOURCE_URL = "https://www.gutenberg.org/cache/epub/108/pg108.txt"

TARGET_WORDS = 650

OVERLAP_PARAGRAPHS = 1


def clean_text(text: str) -> str:
   

    # Standardize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces/tabs inside lines
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce very large blank-line gaps
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_paragraphs(text: str):
    

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    return paragraphs


def count_words(text: str) -> int:
    return len(text.split())


def create_documents(paragraphs):
    

    documents = []

    start_index = 0
    document_number = 1

    while start_index < len(paragraphs):

        current_paragraphs = []
        current_words = 0
        end_index = start_index

        while end_index < len(paragraphs):

            paragraph = paragraphs[end_index]
            paragraph_words = count_words(paragraph)

            if (
                current_words + paragraph_words > TARGET_WORDS
                and current_paragraphs
            ):
                break

            current_paragraphs.append(paragraph)
            current_words += paragraph_words
            end_index += 1

        document_text = "\n\n".join(current_paragraphs)

        document_id = f"DOC_{document_number:03d}"

        document = {
            "document_id": document_id,
            "case_id": CASE_ID,
            "source_title": SOURCE_TITLE,
            "source_url": SOURCE_URL,
            "paragraph_start": start_index + 1,
            "paragraph_end": end_index,
            "word_count": current_words,
            "text": document_text,
        }

        documents.append(document)

        document_number += 1

        
        if end_index >= len(paragraphs):
            break

        start_index = max(
            end_index - OVERLAP_PARAGRAPHS,
            start_index + 1
        )

    return documents


def save_documents(documents):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )


def main():

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw corpus not found: {RAW_PATH}"
        )

    print("Reading raw case corpus...")

    raw_text = RAW_PATH.read_text(
        encoding="utf-8"
    )

    print(
        f"Raw characters: {len(raw_text):,}"
    )

    cleaned_text = clean_text(raw_text)

    paragraphs = split_into_paragraphs(
        cleaned_text
    )

    print(
        f"Paragraphs found: {len(paragraphs)}"
    )

    documents = create_documents(
        paragraphs
    )

    save_documents(
        documents
    )

    print()
    print("Preprocessing complete.")
    print(
        f"Documents created: {len(documents)}"
    )
    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print()
    print("Document summary:")

    for document in documents:

        print(
            document["document_id"],
            "|",
            document["word_count"],
            "words",
            "| paragraphs",
            document["paragraph_start"],
            "-",
            document["paragraph_end"]
        )


if __name__ == "__main__":
    main()