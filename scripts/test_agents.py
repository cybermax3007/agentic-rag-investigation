import json

from backend.agents import (
    AgentEvidenceStore,
    FactCheckerAgent,
    InvestigatorAgent,
)


def main() -> None:
    print("=" * 76)
    print("CASEFILE AI — AGENT SMOKE TEST")
    print("=" * 76)

    store = AgentEvidenceStore()

    print("Hybrid retrieval smoke test:")
    results = store.search(
        "thumbprint blood evidence McFarlane",
        top_k=5,
    )

    for item in results:
        print(
            f"  {item.evidence_id} | "
            f"{item.document_id} | "
            f"{item.status} | "
            f"{item.claim[:100]}"
        )

    print()
    print("Running Investigator Agent...")

    investigator = InvestigatorAgent(
        evidence_store=store
    )

    investigation = investigator.investigate(
        question=(
            "Does the available evidence establish that "
            "John Hector McFarlane murdered Jonas Oldacre?"
        ),
        initial_theory=(
            "McFarlane appears strongly implicated by the "
            "physical and circumstantial evidence."
        ),
        max_iterations=2,
        top_k=7,
    )

    print()
    print("INVESTIGATOR RESULT")
    print("-" * 76)
    print(
        json.dumps(
            investigation.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("Running adversarial Fact-Checker...")

    fact_checker = FactCheckerAgent(
        evidence_store=store
    )

    fact_check = fact_checker.fact_check(
        investigation,
        top_k=8,
    )

    print()
    print("FACT-CHECKER RESULT")
    print("-" * 76)
    print(
        json.dumps(
            fact_check.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
