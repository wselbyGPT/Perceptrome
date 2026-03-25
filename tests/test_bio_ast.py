import unittest

from perceptrome.bio_ast import (
    BioAST,
    CDSNode,
    DomainNode,
    GeneNode,
    GenomeNode,
    KmerNode,
    MicrofeatureNode,
    ORFNode,
    PlasmidNode,
    RegionNode,
    ResidueNode,
    SMENode,
    VirusNode,
    ast_from_flat_genes_payload,
    ast_to_flat_genes_payload,
    build_sme_node,
)


class BioASTTests(unittest.TestCase):
    def test_schema_v2_round_trip_with_hierarchy(self):
        nodes = (
            GenomeNode(canonical_id="genome:1", child_ids=("gene:abc",), start=1, end=1000, strand="+"),
            GeneNode(
                canonical_id="gene:abc",
                gene_id="abc",
                parent_id="genome:1",
                child_ids=("orf:abc:1",),
                start=100,
                end=400,
                strand="+",
                value=0.5,
            ),
            ORFNode(canonical_id="orf:abc:1", parent_id="gene:abc", start=100, end=400, strand="+", frame=0),
            CDSNode(canonical_id="cds:abc:1", parent_id="orf:abc:1", start=115, end=388, strand="+", frame=0),
            RegionNode(canonical_id="region:abc:1", parent_id="cds:abc:1", start=140, end=260),
            DomainNode(canonical_id="domain:abc:1", parent_id="region:abc:1", start=150, end=230),
            SMENode(canonical_id="sme:abc:1", parent_id="domain:abc:1"),
            ResidueNode(canonical_id="residue:abc:1", parent_id="sme:abc:1", start=151, end=151),
            KmerNode(canonical_id="kmer:abc:AAA", parent_id="sme:abc:1", start=151, end=153),
            MicrofeatureNode(canonical_id="microfeature:abc:1", parent_id="sme:abc:1", start=155, end=157),
            PlasmidNode(canonical_id="plasmid:1"),
            VirusNode(canonical_id="virus:1"),
        )
        ast = BioAST(nodes=nodes)

        restored = BioAST.from_dict(ast.to_dict())

        self.assertEqual(restored.schema_version, 4)
        self.assertEqual(len(restored.nodes), len(nodes))
        self.assertEqual(restored.genes[0].gene_id, "abc")
        self.assertEqual(restored.genes[0].parent_id, "genome:1")

    def test_backward_compatible_from_v1_genes_payload(self):
        payload = {
            "genes": [
                {"gene_id": "g1", "value": 1},
                {"gene_id": "g2", "value": 2},
            ]
        }

        ast = BioAST.from_dict(payload)

        self.assertEqual(ast.schema_version, 4)
        self.assertEqual(tuple(g.gene_id for g in ast.genes), ("g1", "g2"))

    def test_flat_adapters_support_legacy_and_new_payloads(self):
        legacy = ast_from_flat_genes_payload({"genes": {"g1": 10, "g2": 20}})
        self.assertEqual(ast_to_flat_genes_payload(legacy), {"genes": {"g1": 10, "g2": 20}})

        modern = ast_from_flat_genes_payload(
            {
                "schema_version": 4,
                "nodes": [{"node_type": "gene", "canonical_id": "g3", "gene_id": "g3", "value": 30}],
            }
        )
        self.assertEqual(ast_to_flat_genes_payload(modern), {"genes": {"g3": 30}})


    def test_sme_payload_validation_and_roundtrip(self):
        sme = SMENode(
            canonical_id="sme:g:1",
            secondary_tag="h",
            motif_family="structural",
            motif_subtype="coiled_coil",
            energetic_evolutionary={
                "folding_energy_estimate": -2.5,
                "phi_bin": -60,
                "psi_bin": -35,
                "conservation_score": 0.9,
                "prion_likelihood": 0.1,
                "variant_sensitivity": 0.4,
            },
        )

        restored = SMENode.from_dict(sme.to_dict())

        self.assertEqual(restored.secondary_tag, "H")
        self.assertEqual(restored.motif_family, "STRUCTURAL")
        self.assertEqual(restored.motif_subtype, "COILED_COIL")
        self.assertEqual(restored.energetic_evolutionary.phi_bin, -60.0)

    def test_build_sme_node_attaches_to_domain(self):
        parent = DomainNode(canonical_id="domain:g:1", child_ids=("existing",))

        updated_parent, sme, children = build_sme_node(
            parent=parent,
            sme_id="sme:g:1",
            residue_window=[(10, 10), 11],
            kmer_window=[(12, 14)],
            secondary_tag="E",
            motif_family="interaction",
            motif_subtype="binding_loop",
            energetic_evolutionary={"conservation_score": 0.5},
        )

        self.assertEqual(updated_parent.child_ids, ("existing", "sme:g:1"))
        self.assertEqual(sme.parent_id, "domain:g:1")
        self.assertEqual(len(children), 3)
        self.assertEqual(children[0].parent_id, "sme:g:1")

    def test_sme_validation_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            SMENode(canonical_id="sme:bad:1", secondary_tag="X")

        with self.assertRaises(ValueError):
            SMENode(
                canonical_id="sme:bad:2",
                energetic_evolutionary={"conservation_score": 1.5},
            )


if __name__ == "__main__":
    unittest.main()
