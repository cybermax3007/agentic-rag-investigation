from backend.agents.evidence_store import AgentEvidenceStore


def main() -> None:
    store = AgentEvidenceStore()

    verified = store.verified_evidence()
    verified_ids = {
        item.evidence_id
        for item in verified
    }

    assert verified, "Verified safety pass returned no evidence."
    assert all(
        item.status == "verified"
        for item in verified
    ), "Safety pass included non-verified evidence."

    # Regression guard for the failure discovered during testing:
    # this verified direct-observation claim must be available to the
    # completeness sweep even if normal retrieval misses it.
    assert "EV_074" in verified_ids, (
        "EV_074 is missing from the verified safety pass."
    )

    ev_074 = next(
        item
        for item in verified
        if item.evidence_id == "EV_074"
    )

    assert ev_074.document_id == "DOC_014"
    assert "darted" in (
        ev_074.claim + " " + ev_074.source_excerpt
    ).lower()

    print("Coverage guard regression test: PASS")
    print(f"Verified evidence available to safety pass: {len(verified)}")
    print(
        "EV_074 available:",
        ev_074.claim,
    )


if __name__ == "__main__":
    main()
