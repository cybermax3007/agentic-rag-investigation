from backend.graph.evidence_graph import (
    EvidenceGraph,
    GRAPH_AUDIT_PATH,
    GRAPH_OUTPUT_PATH,
)


def main() -> None:
    print("=" * 72)
    print("CASEFILE AI — EVIDENCE GRAPH")
    print("=" * 72)

    evidence_graph = EvidenceGraph()
    graph = evidence_graph.build()

    evidence_graph.save()

    audit = evidence_graph.audit()

    print(
        f"Total nodes          : "
        f"{audit['nodes_total']}"
    )
    print(
        f"Total edges          : "
        f"{audit['edges_total']}"
    )

    print("-" * 72)
    print("Node types")

    for node_type, count in sorted(
        audit["node_type_counts"].items()
    ):
        print(
            f"  {node_type:<16} {count}"
        )

    print("-" * 72)
    print("Edge types")

    for edge_type, count in sorted(
        audit["edge_type_counts"].items()
    ):
        print(
            f"  {edge_type:<24} {count}"
        )

    print("-" * 72)
    print("Evidence status")

    for status, count in sorted(
        audit["evidence_status_counts"].items()
    ):
        print(
            f"  {status:<16} {count}"
        )

    print("-" * 72)
    print("Relationship status")

    for status, count in sorted(
        audit[
            "relationship_status_counts"
        ].items()
    ):
        print(
            f"  {status:<16} {count}"
        )

    print("-" * 72)
    print(
        f"Isolated entities    : "
        f"{audit['isolated_entity_count']}"
    )

    if audit["isolated_entities"]:
        print(
            f"  IDs: "
            f"{audit['isolated_entities']}"
        )

    print("=" * 72)
    print(
        f"Graph JSON : {GRAPH_OUTPUT_PATH}"
    )
    print(
        f"Audit JSON : {GRAPH_AUDIT_PATH}"
    )

    # Small smoke test for a core suspect if present.
    suspect = next(
        (
            node_id
            for node_id, data in graph.nodes(
                data=True
            )
            if (
                data.get("node_type") == "entity"
                and data.get("canonical_name")
                == "John Hector McFarlane"
            )
        ),
        None,
    )

    if suspect:
        evidence = evidence_graph.entity_evidence(
            suspect
        )

        relationships = (
            evidence_graph.entity_relationships(
                suspect
            )
        )

        print("-" * 72)
        timeline = evidence_graph.entity_timeline(
            suspect
        )

        print(
            f"McFarlane graph smoke test: "
            f"{len(evidence)} evidence items, "
            f"{len(relationships)} relationships, "
            f"{len(timeline)} timeline events"
        )


if __name__ == "__main__":
    main()
