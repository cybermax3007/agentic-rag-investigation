import json
from pathlib import Path

from backend.models import TimelineCorpus


TIMELINE_PATH = Path("data/processed/timeline_events.json")


def load_timeline(
    path: Path = TIMELINE_PATH,
) -> TimelineCorpus | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return TimelineCorpus.model_validate(json.load(file))
