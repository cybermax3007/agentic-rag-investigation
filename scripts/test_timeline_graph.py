import json
from pathlib import Path

from backend.graph.evidence_graph import EvidenceGraph
from backend.models import CaseEvidenceCorpus, TimelineCorpus


TIMELINE_PATH = Path("data/processed/timeline_events.json")
CASE_PATH = Path("data/processed/canonical_case.json")


def main() -> None:
    if not TIMELINE_PATH.exists():
        raise RuntimeError(
            "Timeline file missing. Run `python -m scripts.build_timeline` first."
        )

    with TIMELINE_PATH.open("r", encoding="utf-8") as file:
        timeline = TimelineCorpus.model_validate(json.load(file))

    with CASE_PATH.open("r", encoding="utf-8") as file:
        case = CaseEvidenceCorpus.model_validate(json.load(file))

    evidence_by_id = {
        item.evidence_id: item
        for item in case.evidence
    }
    evidence_ids = set(evidence_by_id)

    entity_ids = {
        item.entity_id
        for item in case.entities
    }

    assert timeline.timeline_type == "investigation_timeline"
    assert len(timeline.events) >= 6

    assert len(
        {
            event.event_id
            for event in timeline.events
        }
    ) == len(timeline.events)

    assert [
        event.investigation_order
        for event in timeline.events
    ] == list(
        range(1, len(timeline.events) + 1)
    )

    for event in timeline.events:
        assert event.core_evidence_ids, (
            f"{event.event_id} has no core evidence."
        )

        assert set(
            event.core_evidence_ids
        ).issubset(evidence_ids)

        assert set(
            event.context_evidence_ids
        ).issubset(evidence_ids)

        assert not (
            set(event.core_evidence_ids)
            & set(event.context_evidence_ids)
        )

        assert set(
            event.participant_entity_ids
        ).issubset(entity_ids)

        assert set(
            event.location_entity_ids
        ).issubset(entity_ids)

        core_statuses = [
            evidence_by_id[evidence_id].status
            for evidence_id in event.core_evidence_ids
        ]

        # If any direct core evidence is verified, the event itself must not
        # be downgraded because contextual evidence is unverified.
        if "verified" in core_statuses:
            assert event.status == "verified", (
                f"{event.event_id} incorrectly downgraded despite "
                "verified core evidence."
            )

    # Regression guard for the problem discovered during manual testing.
    oldacre_event = next(
        (
            event
            for event in timeline.events
            if "EV_074" in event.core_evidence_ids
        ),
        None,
    )

    assert oldacre_event is not None, (
        "EV_074 must be core evidence for an event."
    )
    assert oldacre_event.status == "verified"
    assert oldacre_event.source_mode == "observed"

    graph = EvidenceGraph()
    graph.build()
    audit = graph.audit()

    event_nodes = audit["node_type_counts"].get(
        "event",
        0,
    )
    time_nodes = audit["node_type_counts"].get(
        "time",
        0,
    )
    sequence_edges = audit["edge_type_counts"].get(
        "investigation_before",
        0,
    )
    support_edges = audit["edge_type_counts"].get(
        "supports_event",
        0,
    )
    context_edges = audit["edge_type_counts"].get(
        "context_for_event",
        0,
    )

    assert event_nodes == len(timeline.events)
    assert sequence_edges == max(
        0,
        len(timeline.events) - 1,
    )
    assert support_edges >= len(timeline.events)

    print("Timeline + graph regression test: PASS")
    print(f"Events: {event_nodes}")
    print(f"Time nodes: {time_nodes}")
    print(
        f"Investigation sequence edges: "
        f"{sequence_edges}"
    )
    print(
        f"Core evidence-to-event edges: "
        f"{support_edges}"
    )
    print(
        f"Context evidence-to-event edges: "
        f"{context_edges}"
    )
    print(
        f"EV_074 event: {oldacre_event.event_id} | "
        f"{oldacre_event.status.upper()} | "
        f"{oldacre_event.source_mode.upper()}"
    )


if __name__ == "__main__":
    main()
