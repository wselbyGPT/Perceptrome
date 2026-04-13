"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const node_test_1 = __importDefault(require("node:test"));
const strict_1 = __importDefault(require("node:assert/strict"));
const schemas_1 = require("../src/features/bio-ast/schemas");
const selection_store_1 = require("../src/features/bio-ast/selection-store");
const bundle = {
    schema: "bio_ast_visualization_bundle_v1",
    accession: "ACC",
    resolved_from: { run_id: "run_1", artifact_id: 1, accession: "ACC", manifest_path: "/tmp/manifest.json", base_dir: "/tmp/bio_ast/ACC" },
    canonical: {
        schema: "bio_ast_canonical_document_v1",
        accession: "ACC",
        export_version: 1,
        schema_version: 3,
        sequence_metadata: {},
        nodes: [
            { canonical_id: "genome:ACC", node_type: "genome", child_ids: ["gene:ACC:1"], metadata: {} },
            { canonical_id: "gene:ACC:1", node_type: "gene", parent_id: "genome:ACC", child_ids: [], start: 10, end: 50, metadata: {} },
        ],
        relationships: [],
    },
    storage_map: {
        schema: "bio_ast_storage_map_v1",
        accession: "ACC",
        export_version: 1,
        sequence_length: 120,
        topology: {},
        tracks: [{ track_id: "track:0", track_index: 0, strand: "unstranded", node_type: "gene", lane_count: 1, segment_count: 1, segment_range: [0, 1] }],
        coordinate_segments: [{ segment_id: "gene:ACC:1", node_id: "gene:ACC:1", node_type: "gene", strand: "unstranded", track_id: "track:0", track_index: 0, lane_index: 0, start: 10, end: 50, length: 41 }],
    },
    tree: {
        schema: "bio_ast_tree_v1",
        accession: "ACC",
        export_version: 1,
        node_count: 2,
        sequence_metadata: {},
        roots: ["genome:ACC"],
        hierarchy: [{ id: "genome:ACC", label: "genome", node_type: "genome", span: {}, metadata: {}, children: [{ id: "gene:ACC:1", label: "gene", node_type: "gene", parent_id: "genome:ACC", span: {}, metadata: {}, children: [] }] }],
    },
    graph: {
        schema: "bio_ast_graph_v1",
        accession: "ACC",
        export_version: 1,
        node_count: 2,
        edge_count: 1,
        hierarchy_edge_count: 1,
        semantic_edge_count: 0,
        sequence_metadata: {},
        nodes: [
            { id: "genome:ACC", label: "genome", node_type: "genome", index: 0, span: {}, metadata: {} },
            { id: "gene:ACC:1", label: "gene", node_type: "gene", parent_id: "genome:ACC", index: 1, span: {}, metadata: {} },
        ],
        edges: [{ id: "edge-1", source: "genome:ACC", target: "gene:ACC:1", source_index: 0, target_index: 1, relation: "contains", edge_kind: "contains", relation_type: "hierarchy", metadata: {} }],
        hierarchy_edges: [{ id: "edge-1", source: "genome:ACC", target: "gene:ACC:1", source_index: 0, target_index: 1, relation: "contains", edge_kind: "contains", relation_type: "hierarchy", metadata: {} }],
        semantic_edges: [],
    },
    summary: {},
};
(0, node_test_1.default)("selection store synchronizes node and edge focus across panes", () => {
    const store = (0, selection_store_1.createBioAstSelectionStore)();
    const states = [];
    const unsubscribe = store.subscribe((state) => {
        states.push(`${state.selectedNodeId ?? "-"}|${state.hoveredNodeId ?? "-"}|${state.selectedEdgeId ?? "-"}|${state.hoveredEdgeId ?? "-"}`);
    });
    store.selectNode("gene:ACC:1");
    store.hoverNode("gene:ACC:1");
    store.selectEdge("edge-1");
    store.hoverEdge("edge-1");
    unsubscribe();
    strict_1.default.deepEqual(states.slice(1), [
        "gene:ACC:1|-|-|-",
        "gene:ACC:1|gene:ACC:1|-|-",
        "-|gene:ACC:1|edge-1|-",
        "-|gene:ACC:1|edge-1|edge-1",
    ]);
});
(0, node_test_1.default)("bundle schema validation rejects cross-payload node mismatches", () => {
    const parsed = schemas_1.bioAstVisualizationBundleSchema.parse(bundle);
    strict_1.default.equal(parsed.graph.nodes[1].id, "gene:ACC:1");
    const invalid = structuredClone(bundle);
    invalid.storage_map.coordinate_segments[0].node_id = "missing:node";
    strict_1.default.throws(() => schemas_1.bioAstVisualizationBundleSchema.parse(invalid), /unknown node missing:node/);
});
