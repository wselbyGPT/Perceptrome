import unittest

from perceptrome.jobs.lineage import build_lineage_adjacency
from perceptrome.jobs.manifest_schema import empty_run_manifest, normalize_run_manifest


class ManifestLineageTests(unittest.TestCase):
    def test_manifest_schema_defaults_include_lineage(self):
        manifest = empty_run_manifest(
            run_kind="generate_plasmid",
            config_path="config.yml",
            config_hash=None,
            run_id="r1",
            created_at="2025-01-01T00:00:00Z",
            git_sha=None,
        )
        self.assertEqual(manifest["run"]["parents"], [])
        self.assertEqual(manifest["run"]["children"], [])

        normalized = normalize_run_manifest({"manifest_type": "run_manifest", "run": {"id": "r2"}, "artifacts": [{"id": "a", "path": "x"}]})
        self.assertEqual(normalized["run"]["parents"], [])
        self.assertEqual(normalized["run"]["children"], [])
        self.assertEqual(normalized["artifacts"][0]["parents"], [])

    def test_lineage_adjacency_from_run_and_artifact_links(self):
        manifest = {
            "manifest_type": "run_manifest",
            "run": {
                "id": "r3",
                "parents": [{"artifact_id": "checkpoint", "path": "model/latest.pt", "relation": "consumed.checkpoint"}],
                "children": [{"artifact_id": "generated", "path": "outputs/out.fasta", "relation": "produced.generated_sequence"}],
            },
            "artifacts": [
                {
                    "id": "generated",
                    "path": "outputs/out.fasta",
                    "role": "generated.sequence",
                    "parents": [{"artifact_id": "checkpoint", "path": "model/latest.pt", "relation": "consumed.checkpoint"}],
                }
            ],
        }

        graph = build_lineage_adjacency(manifest)
        edges = {(edge["source"], edge["target"], edge["relation"]) for edge in graph["edges"]}

        self.assertIn(("artifact:checkpoint", "run:r3", "consumed.checkpoint"), edges)
        self.assertIn(("run:r3", "artifact:generated", "produced.generated_sequence"), edges)
        self.assertIn(("artifact:checkpoint", "artifact:generated", "consumed.checkpoint"), edges)


if __name__ == "__main__":
    unittest.main()
