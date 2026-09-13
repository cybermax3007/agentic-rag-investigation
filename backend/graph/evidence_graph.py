import json
from collections import Counter
from pathlib import Path

import networkx as nx

from backend.models import CaseEvidenceCorpus, CorpusDocument


CANONICAL_CASE_PATH = Path("data/processed/canonical_case.json")
DOCUMENTS_PATH = Path("data/processed/documents.json")
GRAPH_OUTPUT_PATH = Path("data/processed/evidence_graph.json")
GRAPH_AUDIT_PATH = Path("data/processed/evidence_graph_audit.json")


class EvidenceGraph:
    """
    Typed evidence graph for CaseFile AI.

    Node types:
      - entity
      - evidence
      - document
      - hypothesis

    Edge types:
      - mentioned_in
      - extracted_from
      - about_entity
      - supports
      - contradicts
      - extracted_relationship

    MultiDiGraph is used because multiple evidence/relationship edges may
    connect the same pair of nodes.
    """

    def __init__(
        self,
        canonical_path: Path = CANONICAL_CASE_PATH,
        documents_path: Path = DOCUMENTS_PATH,
    ):
        self.canonical_path = canonical_path
        self.documents_path = documents_path
        self.graph = nx.MultiDiGraph()

        self.case: CaseEvidenceCorpus | None = None
        self.documents: dict[str, CorpusDocument] = {}

    def load(self) -> None:
        if not self.canonical_path.exists():
            raise FileNotFoundError(
                f"Canonical case file not found: {self.canonical_path}"
            )

        if not self.documents_path.exists():
            raise FileNotFoundError(
                f"Documents file not found: {self.documents_path}"
            )

        with self.canonical_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.case = CaseEvidenceCorpus.model_validate(
                json.load(file)
            )

        with self.documents_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_documents = json.load(file)

        self.documents = {
            item["document_id"]: CorpusDocument.model_validate(item)
            for item in raw_documents
        }

    @staticmethod
    def _hypothesis_node_id(label: str) -> str:
        normalized = "_".join(
            label.lower().strip().split()
        )
        normalized = "".join(
            char
            for char in normalized
            if char.isalnum() or char == "_"
        )
        return f"HYP_{normalized}"

    def build(self) -> nx.MultiDiGraph:
        if self.case is None:
            self.load()

        assert self.case is not None

        self.graph.clear()

        # -------------------------------------------------------------
        # Document nodes
        # -------------------------------------------------------------
        for document_id, document in sorted(
            self.documents.items()
        ):
            self.graph.add_node(
                document_id,
                node_type="document",
                label=document_id,
                source_title=document.source_title,
                source_url=document.source_url,
                paragraph_start=document.paragraph_start,
                paragraph_end=document.paragraph_end,
                word_count=document.word_count,
                text=document.text,
            )

        # -------------------------------------------------------------
        # Entity nodes
        # -------------------------------------------------------------
        for entity in self.case.entities:
            self.graph.add_node(
                entity.entity_id,
                node_type="entity",
                label=entity.canonical_name,
                canonical_name=entity.canonical_name,
                entity_type=entity.entity_type,
                aliases=entity.aliases,
                document_ids=entity.document_ids,
                mention_count=entity.mention_count,
                description=entity.description,
            )

            for document_id in entity.document_ids:
                if document_id in self.graph:
                    self.graph.add_edge(
                        entity.entity_id,
                        document_id,
                        key=f"MENTION_{entity.entity_id}_{document_id}",
                        edge_type="mentioned_in",
                    )

        # -------------------------------------------------------------
        # Evidence nodes
        # -------------------------------------------------------------
        for evidence in self.case.evidence:
            self.graph.add_node(
                evidence.evidence_id,
                node_type="evidence",
                label=evidence.claim,
                claim=evidence.claim,
                evidence_type=evidence.evidence_type,
                status=evidence.status,
                source_speaker=evidence.source_speaker,
                source_excerpt=evidence.source_excerpt,
                document_id=evidence.document_id,
                is_grounded=evidence.is_grounded,
                supports=evidence.supports,
                contradicts=evidence.contradicts,
            )

            if evidence.document_id in self.graph:
                self.graph.add_edge(
                    evidence.evidence_id,
                    evidence.document_id,
                    key=f"SOURCE_{evidence.evidence_id}",
                    edge_type="extracted_from",
                )

            for entity_id in evidence.entity_ids:
                if entity_id in self.graph:
                    self.graph.add_edge(
                        evidence.evidence_id,
                        entity_id,
                        key=(
                            f"ABOUT_{evidence.evidence_id}_"
                            f"{entity_id}"
                        ),
                        edge_type="about_entity",
                    )

            for hypothesis in evidence.supports:
                hypothesis_id = self._hypothesis_node_id(
                    hypothesis
                )

                if hypothesis_id not in self.graph:
                    self.graph.add_node(
                        hypothesis_id,
                        node_type="hypothesis",
                        label=hypothesis,
                        hypothesis=hypothesis,
                    )

                self.graph.add_edge(
                    evidence.evidence_id,
                    hypothesis_id,
                    key=(
                        f"SUPPORT_{evidence.evidence_id}_"
                        f"{hypothesis_id}"
                    ),
                    edge_type="supports",
                    status=evidence.status,
                )

            for hypothesis in evidence.contradicts:
                hypothesis_id = self._hypothesis_node_id(
                    hypothesis
                )

                if hypothesis_id not in self.graph:
                    self.graph.add_node(
                        hypothesis_id,
                        node_type="hypothesis",
                        label=hypothesis,
                        hypothesis=hypothesis,
                    )

                self.graph.add_edge(
                    evidence.evidence_id,
                    hypothesis_id,
                    key=(
                        f"CONTRADICT_{evidence.evidence_id}_"
                        f"{hypothesis_id}"
                    ),
                    edge_type="contradicts",
                    status=evidence.status,
                )

        # -------------------------------------------------------------
        # Explicit extracted relationships
        # -------------------------------------------------------------
        for relation in self.case.relationships:
            if (
                relation.source_id not in self.graph
                or relation.target_id not in self.graph
            ):
                continue

            self.graph.add_edge(
                relation.source_id,
                relation.target_id,
                key=relation.relationship_id,
                edge_type="extracted_relationship",
                relationship_id=relation.relationship_id,
                relation=relation.relation,
                status=relation.status,
                source_speaker=relation.source_speaker,
                source_excerpt=relation.source_excerpt,
                is_grounded=relation.is_grounded,
                document_id=relation.document_id,
            )

        return self.graph

    def entity_evidence(
        self,
        entity_id: str,
    ) -> list[dict]:
        if entity_id not in self.graph:
            return []

        evidence_items = []

        for predecessor in self.graph.predecessors(
            entity_id
        ):
            node = self.graph.nodes[predecessor]

            if node.get("node_type") != "evidence":
                continue

            edge_bundle = self.graph.get_edge_data(
                predecessor,
                entity_id,
            )

            if not edge_bundle:
                continue

            if any(
                edge.get("edge_type") == "about_entity"
                for edge in edge_bundle.values()
            ):
                evidence_items.append(
                    {
                        "evidence_id": predecessor,
                        "claim": node.get("claim"),
                        "status": node.get("status"),
                        "evidence_type": node.get(
                            "evidence_type"
                        ),
                        "document_id": node.get(
                            "document_id"
                        ),
                        "source_excerpt": node.get(
                            "source_excerpt"
                        ),
                    }
                )

        return sorted(
            evidence_items,
            key=lambda item: item["evidence_id"],
        )

    def entity_relationships(
        self,
        entity_id: str,
    ) -> list[dict]:
        if entity_id not in self.graph:
            return []

        rows = []

        for source, target, key, data in (
            self.graph.edges(
                entity_id,
                keys=True,
                data=True,
            )
        ):
            if (
                data.get("edge_type")
                != "extracted_relationship"
            ):
                continue

            rows.append(
                {
                    "relationship_id": key,
                    "direction": "outgoing",
                    "source_id": source,
                    "source_name": self.graph.nodes[
                        source
                    ].get("label"),
                    "target_id": target,
                    "target_name": self.graph.nodes[
                        target
                    ].get("label"),
                    "relation": data.get("relation"),
                    "status": data.get("status"),
                    "document_id": data.get(
                        "document_id"
                    ),
                    "source_excerpt": data.get(
                        "source_excerpt"
                    ),
                }
            )

        for source, target, key, data in (
            self.graph.in_edges(
                entity_id,
                keys=True,
                data=True,
            )
        ):
            if (
                data.get("edge_type")
                != "extracted_relationship"
            ):
                continue

            rows.append(
                {
                    "relationship_id": key,
                    "direction": "incoming",
                    "source_id": source,
                    "source_name": self.graph.nodes[
                        source
                    ].get("label"),
                    "target_id": target,
                    "target_name": self.graph.nodes[
                        target
                    ].get("label"),
                    "relation": data.get("relation"),
                    "status": data.get("status"),
                    "document_id": data.get(
                        "document_id"
                    ),
                    "source_excerpt": data.get(
                        "source_excerpt"
                    ),
                }
            )

        return sorted(
            rows,
            key=lambda item: item["relationship_id"],
        )

    def subgraph_for_entity(
        self,
        entity_id: str,
        evidence_limit: int = 20,
    ) -> dict:
        if entity_id not in self.graph:
            raise KeyError(
                f"Unknown entity ID: {entity_id}"
            )

        node_ids = {entity_id}

        evidence = self.entity_evidence(
            entity_id
        )[:evidence_limit]

        for item in evidence:
            node_ids.add(item["evidence_id"])

            document_id = item.get("document_id")
            if document_id:
                node_ids.add(document_id)

        for relation in self.entity_relationships(
            entity_id
        ):
            node_ids.add(relation["source_id"])
            node_ids.add(relation["target_id"])

        subgraph = self.graph.subgraph(node_ids)

        return nx.node_link_data(
            subgraph,
            edges="edges",
        )

    def audit(self) -> dict:
        node_type_counts = Counter(
            data.get("node_type", "unknown")
            for _, data in self.graph.nodes(
                data=True
            )
        )

        edge_type_counts = Counter(
            data.get("edge_type", "unknown")
            for _, _, _, data in self.graph.edges(
                keys=True,
                data=True,
            )
        )

        evidence_status_counts = Counter(
            data.get("status")
            for _, data in self.graph.nodes(
                data=True
            )
            if data.get("node_type") == "evidence"
        )

        relationship_status_counts = Counter(
            data.get("status")
            for _, _, _, data in self.graph.edges(
                keys=True,
                data=True,
            )
            if (
                data.get("edge_type")
                == "extracted_relationship"
            )
        )

        isolated_entities = [
            node_id
            for node_id, data in self.graph.nodes(
                data=True
            )
            if (
                data.get("node_type") == "entity"
                and self.graph.degree(node_id) == 0
            )
        ]

        return {
            "nodes_total": self.graph.number_of_nodes(),
            "edges_total": self.graph.number_of_edges(),
            "node_type_counts": dict(
                node_type_counts
            ),
            "edge_type_counts": dict(
                edge_type_counts
            ),
            "evidence_status_counts": dict(
                evidence_status_counts
            ),
            "relationship_status_counts": dict(
                relationship_status_counts
            ),
            "isolated_entities": isolated_entities,
            "isolated_entity_count": len(
                isolated_entities
            ),
        }

    def save(
        self,
        graph_path: Path = GRAPH_OUTPUT_PATH,
        audit_path: Path = GRAPH_AUDIT_PATH,
    ) -> None:
        graph_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        graph_data = nx.node_link_data(
            self.graph,
            edges="edges",
        )

        with graph_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                graph_data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        with audit_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.audit(),
                file,
                indent=2,
                ensure_ascii=False,
            )
