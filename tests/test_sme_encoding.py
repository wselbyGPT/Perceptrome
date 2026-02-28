import unittest

from perceptrome.bio_ast import EnergeticEvolutionaryPayload, SMENode
from tests.fixtures.bio_ast_regression_fixtures import SECONDARY_TAG_SET, SME_MOTIF_FIXTURES


class SMEEncodingTests(unittest.TestCase):
    def test_secondary_tag_normalization_for_mixed_tags(self):
        normalized = [SMENode(canonical_id=f"sme:tag:{i}", secondary_tag=tag).secondary_tag for i, tag in enumerate(SECONDARY_TAG_SET, start=1)]
        self.assertEqual(normalized, ["H", "E", "C", "T", "G", "I"])

    def test_energetic_evolutionary_population_round_trip(self):
        fixtures = [
            SMENode(canonical_id=f"sme:pop:{idx}", energetic_evolutionary=item["energetic_evolutionary"])
            for idx, item in enumerate(SME_MOTIF_FIXTURES, start=1)
        ]
        payloads = [node.to_dict()["energetic_evolutionary"] for node in fixtures]

        restored = [EnergeticEvolutionaryPayload.from_dict(item) for item in payloads]

        self.assertAlmostEqual(restored[0].folding_energy_estimate, -8.5)
        self.assertAlmostEqual(restored[1].conservation_score, 0.91)

    def test_sme_edge_case_validation(self):
        boundary = SMENode(
            canonical_id="sme:boundary:1",
            energetic_evolutionary={
                "phi_bin": -180,
                "psi_bin": 180,
                "conservation_score": 0.0,
                "prion_likelihood": 1.0,
                "variant_sensitivity": 1.0,
            },
        )
        self.assertEqual(boundary.energetic_evolutionary.phi_bin, -180.0)
        self.assertEqual(boundary.energetic_evolutionary.psi_bin, 180.0)

        with self.assertRaises(ValueError):
            SMENode(canonical_id="sme:bad:1", secondary_tag="Z")

        with self.assertRaises(ValueError):
            SMENode(canonical_id="sme:bad:2", energetic_evolutionary={"psi_bin": 181})


if __name__ == "__main__":
    unittest.main()
