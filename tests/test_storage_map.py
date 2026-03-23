from perceptrome.bio_ast import BioAST, CDSNode, DomainNode, GeneNode, GenomeNode, RegionNode
from perceptrome.encoding.storage_map import STORAGE_MAP_SCHEMA, build_storage_map_payload


def test_storage_map_segment_order_is_deterministic():
    ast = BioAST(
        nodes=(
            GenomeNode(canonical_id="genome:acc", start=1, end=200),
            DomainNode(canonical_id="domain:z", parent_id="gene:a", start=70, end=80, strand="+"),
            GeneNode(canonical_id="gene:b", parent_id="genome:acc", start=40, end=90, strand="+"),
            DomainNode(canonical_id="domain:a", parent_id="gene:a", start=70, end=80, strand="+"),
            GeneNode(canonical_id="gene:a", parent_id="genome:acc", start=10, end=50, strand="+"),
        )
    )

    payload = build_storage_map_payload(ast, 200, accession="ACC")

    assert payload["schema"] == STORAGE_MAP_SCHEMA
    assert [segment["node_id"] for segment in payload["coordinate_segments"]] == [
        "gene:a",
        "gene:b",
        "domain:a",
        "domain:z",
    ]


def test_storage_map_packs_overlaps_into_stable_lanes():
    ast = BioAST(
        nodes=(
            GenomeNode(canonical_id="genome:acc", start=1, end=200),
            GeneNode(canonical_id="gene:a", parent_id="genome:acc", start=1, end=50, strand="+"),
            GeneNode(canonical_id="gene:b", parent_id="genome:acc", start=20, end=40, strand="+"),
            GeneNode(canonical_id="gene:c", parent_id="genome:acc", start=51, end=70, strand="+"),
        )
    )

    payload = build_storage_map_payload(ast, 200)
    lanes = {segment["node_id"]: segment["lane_index"] for segment in payload["coordinate_segments"]}
    track = payload["tracks"][0]

    assert lanes == {"gene:a": 0, "gene:b": 1, "gene:c": 0}
    assert track["lane_count"] == 2


def test_storage_map_assigns_tracks_by_strand_and_node_type():
    ast = BioAST(
        nodes=(
            GenomeNode(canonical_id="genome:acc", start=1, end=200),
            GeneNode(canonical_id="gene:plus", parent_id="genome:acc", start=1, end=20, strand="+"),
            GeneNode(canonical_id="gene:minus", parent_id="genome:acc", start=30, end=60, strand="-"),
            RegionNode(canonical_id="region:minus", parent_id="gene:minus", start=35, end=45, strand="-"),
            CDSNode(canonical_id="cds:none", parent_id="gene:plus", start=5, end=15),
        )
    )

    payload = build_storage_map_payload(ast, 200)
    tracks = {(track["strand"], track["node_type"]): track["track_index"] for track in payload["tracks"]}

    assert ("+", "gene") in tracks
    assert ("-", "gene") in tracks
    assert ("-", "region") in tracks
    assert ("unstranded", "cds") in tracks

    segment_tracks = {
        segment["node_id"]: (segment["strand"], segment["node_type"], segment["track_index"])
        for segment in payload["coordinate_segments"]
    }
    assert segment_tracks["gene:plus"][:2] == ("+", "gene")
    assert segment_tracks["gene:minus"][:2] == ("-", "gene")
    assert segment_tracks["region:minus"][:2] == ("-", "region")
    assert segment_tracks["cds:none"][:2] == ("unstranded", "cds")
