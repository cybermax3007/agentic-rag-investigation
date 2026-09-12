from pathlib import Path
import requests


URL = "https://www.gutenberg.org/cache/epub/108/pg108.txt"

START_MARKER = "THE ADVENTURE OF THE NORWOOD BUILDER"
END_MARKER = "THE ADVENTURE OF THE DANCING MEN"


def download_case():
    print("Downloading The Return of Sherlock Holmes...")

    response = requests.get(URL, timeout=30)
    response.raise_for_status()

    full_text = response.text

    start = full_text.find(START_MARKER)

    if start == -1:
        raise ValueError(
            "Could not find the beginning of The Adventure of the Norwood Builder."
        )

    end = full_text.find(
        END_MARKER,
        start + len(START_MARKER)
    )

    if end == -1:
        raise ValueError(
            "Could not find the end of The Adventure of the Norwood Builder."
        )

    case_text = full_text[start:end].strip()

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "norwood_builder.txt"

    output_path.write_text(
        case_text,
        encoding="utf-8"
    )

    print("Download complete.")
    print(f"Saved to: {output_path}")
    print(f"Characters: {len(case_text):,}")


if __name__ == "__main__":
    download_case()