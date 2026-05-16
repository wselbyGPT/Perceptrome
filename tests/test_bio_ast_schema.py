import unittest

from perceptrome.bio_ast import (
    BioAST,
    CONTAINMENT_EDGE_KIND,
    DomainNode,
    GeneNode,
    RelationshipEdge,
    SMENode,
    SequenceMetadata,
    build_sme_node,
    migrate_bio_ast_payload,
)
from perceptrome.encoding.bio_ast_builder import BioASTBuilder
from tests.fixtures.bio_ast_regression_fixtures import SME_MOTIF_FIXTURES, SYNTHETIC_FASTA_SEQUENCE


class BioASTSchemaTests(unittest.TestCase):
    def test_serialization_deserialization_round_trip(self):
        motif = SME_MOTIF_FIXTURES[0]
        nodes = (
            GeneNode(canonical_id="gene:motif:1", gene_id="motif1", child_ids=("domain:motif:1",), start=1, end=120, strand="+"),
            DomainNode(canonical_id="domain:motif:1", parent_id="gene:motif:1", child_ids=("sme:motif:1",), start=20, end=90),
            SMENode(
                canonical_id="sme:motif:1",
                parent_id="domain:motif:1",
                start=25,
                end=55,
                secondary_tag=motif["secondary_tag"],
                motif_family=motif["motif_family"],
                motif_subtype=motif["motif_subtype"],
                energetic_evolutionary=motif["energetic_evolutionary"],
                metadata={"motif_name": motif["motif_name"]},
            ),
        )
        ast = BioAST(
            nodes=nodes,
            sequence_metadata=SequenceMetadata(accession="motif", length=120, topology="linear", molecule_type="DNA", source_format="fixture", checksum="abc"),
            relationships=(RelationshipEdge(source_id="gene:motif:1", target_id="sme:motif:1", kind="overlaps"),),
        )

        restored = BioAST.from_dict(ast.to_dict())

        self.assertEqual(restored.schema_version, 4)
        self.assertEqual(restored.sequence_metadata.topology, "linear")
        self.assertEqual(len(restored.nodes), 3)
        self.assertEqual(len(restored.containment_edges), 2)
        self.assertEqual(restored.semantic_edges[0].kind, "overlaps")
        restored_sme = next(node for node in restored.nodes if isinstance(node, SMENode))
        self.assertEqual(restored_sme.metadata.get("motif_name"), "helix-entry")
        self.assertEqual(restored_sme.secondary_tag, "H")

    def test_schema_version_migration_from_v1(self):
        payload = {"schema_version": 1, "genes": [{"gene_id": "gA", "value": 1.0}, {"gene_id": "gB", "value": 2.0}]}
        migrated = migrate_bio_ast_payload(payload)
        ast = BioAST.from_dict(payload)

        self.assertEqual(migrated["schema_version"], 4)
        self.assertIn("sequence_metadata", migrated)
        self.assertIn("edges", migrated)
        self.assertEqual(ast.schema_version, 4)
        self.assertEqual(tuple(g.gene_id for g in ast.genes), ("gA", "gB"))
        self.assertEqual(len(ast.nodes), 2)

    def test_parent_child_integrity_for_sme_subtree(self):
        parent = DomainNode(canonical_id="domain:int:1", child_ids=("existing",), start=10, end=80)
        updated_parent, sme, children = build_sme_node(
            parent=parent,
            sme_id="sme:int:1",
            residue_window=[(20, 20), 21],
            kmer_window=[(22, 24)],
            secondary_tag=SME_MOTIF_FIXTURES[1]["secondary_tag"],
            motif_family=SME_MOTIF_FIXTURES[1]["motif_family"],
            motif_subtype=SME_MOTIF_FIXTURES[1]["motif_subtype"],
            energetic_evolutionary=SME_MOTIF_FIXTURES[1]["energetic_evolutionary"],
            metadata={"motif_name": SME_MOTIF_FIXTURES[1]["motif_name"]},
        )

        self.assertIn(sme.canonical_id, updated_parent.child_ids)
        self.assertEqual(sme.parent_id, updated_parent.canonical_id)
        self.assertTrue(all(child.parent_id == sme.canonical_id for child in children))

    def test_edge_kind_validation_and_legacy_alias_migration(self):
        legacy = RelationshipEdge.from_dict({"source_gene_id": "g1", "target_gene_id": "g2", "relation": "depends_on"})
        self.assertEqual(legacy.kind, "regulates")
        self.assertEqual(legacy.source_gene_id, "g1")
        support_alias = RelationshipEdge(source_id="cds1", target_id="protein1", kind="supports")
        self.assertEqual(support_alias.kind, "encodes")
        with self.assertRaises(ValueError):
            RelationshipEdge(source_id="g1", target_id="g2", kind="unsupported")

    def test_builder_emits_typed_document_and_roundtrip(self):
        built = BioASTBuilder().build(sequence=SYNTHETIC_FASTA_SEQUENCE, accession="ROUND1", source_format="fasta")
        payload = built.ast.to_dict()
        restored = BioAST.from_dict(payload)

        self.assertEqual(payload["sequence_metadata"]["source_format"], "fasta")
        self.assertEqual(payload["sequence_metadata"]["checksum"], restored.sequence_metadata.checksum)
        self.assertEqual([n["canonical_id"] for n in payload["nodes"]], [node.canonical_id for node in restored.nodes])
        self.assertTrue(all(edge["kind"] == CONTAINMENT_EDGE_KIND or edge["kind"] in {"regulates", "overlaps", "adjacent_to", "homologous_to", "promoter_of", "operator_of", "rbs_for", "terminates", "part_of_transcript_unit", "part_of_operon", "part_of_module", "encodes", "produces_transcript", "produces_protein", "upstream_of", "downstream_of", "same_strand_as", "opposite_strand_of"} for edge in payload["edges"]))


if __name__ == "__main__":
    unittest.main()
