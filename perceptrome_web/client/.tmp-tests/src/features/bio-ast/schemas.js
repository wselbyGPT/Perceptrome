"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.bioAstVisualizationBundleSchema = exports.bioAstGraphSchema = exports.bioAstGraphEdgeSchema = exports.bioAstGraphNodeSchema = exports.bioAstTreeSchema = exports.bioAstTreeNodeSchema = exports.bioAstStorageMapSchema = exports.bioAstStorageSegmentSchema = exports.bioAstStorageTrackSchema = exports.bioAstCanonicalSchema = exports.bioAstCanonicalNodeSchema = void 0;
const zod_1 = require("zod");
exports.bioAstCanonicalNodeSchema = zod_1.z.object({
    canonical_id: zod_1.z.string(),
    node_type: zod_1.z.string(),
    parent_id: zod_1.z.string().nullable().optional(),
    child_ids: zod_1.z.array(zod_1.z.string()).default([]),
    start: zod_1.z.number().int().nullable().optional(),
    end: zod_1.z.number().int().nullable().optional(),
    strand: zod_1.z.string().nullable().optional(),
    frame: zod_1.z.number().int().nullable().optional(),
    metadata: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
});
exports.bioAstCanonicalSchema = zod_1.z.object({
    schema: zod_1.z.literal("bio_ast_canonical_document_v1"),
    accession: zod_1.z.string().nullable().optional(),
    export_version: zod_1.z.number().int().optional(),
    schema_version: zod_1.z.number().int().optional(),
    sequence_metadata: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
    nodes: zod_1.z.array(exports.bioAstCanonicalNodeSchema),
    relationships: zod_1.z.array(zod_1.z.record(zod_1.z.string(), zod_1.z.unknown())).default([]),
});
exports.bioAstStorageTrackSchema = zod_1.z.object({
    track_id: zod_1.z.string(),
    track_index: zod_1.z.number().int(),
    strand: zod_1.z.string(),
    node_type: zod_1.z.string(),
    lane_count: zod_1.z.number().int(),
    segment_count: zod_1.z.number().int(),
    segment_range: zod_1.z.array(zod_1.z.number().int()).length(2),
});
exports.bioAstStorageSegmentSchema = zod_1.z.object({
    segment_id: zod_1.z.string(),
    node_id: zod_1.z.string(),
    parent_id: zod_1.z.string().nullable().optional(),
    node_type: zod_1.z.string(),
    strand: zod_1.z.string(),
    track_id: zod_1.z.string(),
    track_index: zod_1.z.number().int(),
    lane_index: zod_1.z.number().int(),
    start: zod_1.z.number().int(),
    end: zod_1.z.number().int(),
    length: zod_1.z.number().int(),
});
exports.bioAstStorageMapSchema = zod_1.z.object({
    schema: zod_1.z.literal("bio_ast_storage_map_v1"),
    accession: zod_1.z.string().nullable().optional(),
    export_version: zod_1.z.number().int().optional(),
    sequence_length: zod_1.z.number().int(),
    topology: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
    tracks: zod_1.z.array(exports.bioAstStorageTrackSchema),
    coordinate_segments: zod_1.z.array(exports.bioAstStorageSegmentSchema),
});
exports.bioAstTreeNodeSchema = zod_1.z.lazy(() => zod_1.z.object({
    id: zod_1.z.string(),
    label: zod_1.z.string(),
    node_type: zod_1.z.string(),
    parent_id: zod_1.z.string().nullable().optional(),
    span: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
    metadata: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
    coordinates: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).nullable().optional(),
    children: zod_1.z.array(exports.bioAstTreeNodeSchema).default([]),
}));
exports.bioAstTreeSchema = zod_1.z.object({
    schema: zod_1.z.literal("bio_ast_tree_v1"),
    accession: zod_1.z.string().nullable().optional(),
    export_version: zod_1.z.number().int().optional(),
    node_count: zod_1.z.number().int(),
    sequence_metadata: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
    roots: zod_1.z.array(zod_1.z.string()),
    hierarchy: zod_1.z.array(exports.bioAstTreeNodeSchema),
});
exports.bioAstGraphNodeSchema = zod_1.z.object({
    id: zod_1.z.string(),
    label: zod_1.z.string(),
    node_type: zod_1.z.string(),
    parent_id: zod_1.z.string().nullable().optional(),
    span: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
    metadata: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
    coordinates: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).nullable().optional(),
    index: zod_1.z.number().int(),
});
exports.bioAstGraphEdgeSchema = zod_1.z.object({
    id: zod_1.z.string(),
    source: zod_1.z.string(),
    target: zod_1.z.string(),
    source_index: zod_1.z.number().int(),
    target_index: zod_1.z.number().int(),
    relation: zod_1.z.string(),
    edge_kind: zod_1.z.string(),
    relation_type: zod_1.z.string(),
    evidence: zod_1.z.string().nullable().optional(),
    metadata: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
});
exports.bioAstGraphSchema = zod_1.z.object({
    schema: zod_1.z.literal("bio_ast_graph_v1"),
    accession: zod_1.z.string().nullable().optional(),
    export_version: zod_1.z.number().int().optional(),
    node_count: zod_1.z.number().int(),
    edge_count: zod_1.z.number().int(),
    hierarchy_edge_count: zod_1.z.number().int(),
    semantic_edge_count: zod_1.z.number().int(),
    sequence_metadata: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
    nodes: zod_1.z.array(exports.bioAstGraphNodeSchema),
    edges: zod_1.z.array(exports.bioAstGraphEdgeSchema),
    hierarchy_edges: zod_1.z.array(exports.bioAstGraphEdgeSchema),
    semantic_edges: zod_1.z.array(exports.bioAstGraphEdgeSchema),
});
exports.bioAstVisualizationBundleSchema = zod_1.z
    .object({
    schema: zod_1.z.literal("bio_ast_visualization_bundle_v1"),
    accession: zod_1.z.string().nullable().optional(),
    resolved_from: zod_1.z.object({
        run_id: zod_1.z.string(),
        artifact_id: zod_1.z.number().int().nullable().optional(),
        accession: zod_1.z.string().nullable().optional(),
        manifest_path: zod_1.z.string().nullable().optional(),
        base_dir: zod_1.z.string(),
    }),
    canonical: exports.bioAstCanonicalSchema,
    storage_map: exports.bioAstStorageMapSchema,
    tree: exports.bioAstTreeSchema,
    graph: exports.bioAstGraphSchema,
    summary: zod_1.z.record(zod_1.z.string(), zod_1.z.unknown()).default({}),
})
    .superRefine((bundle, ctx) => {
    const canonicalIds = new Set(bundle.canonical.nodes.map((node) => node.canonical_id));
    const treeIds = new Set();
    const visit = (nodes) => {
        for (const node of nodes) {
            treeIds.add(node.id);
            visit(node.children);
        }
    };
    visit(bundle.tree.hierarchy);
    for (const nodeId of treeIds) {
        if (!canonicalIds.has(nodeId)) {
            ctx.addIssue({ code: zod_1.z.ZodIssueCode.custom, message: `Tree node ${nodeId} is missing from canonical payload` });
        }
    }
    for (const segment of bundle.storage_map.coordinate_segments) {
        if (!canonicalIds.has(segment.node_id)) {
            ctx.addIssue({ code: zod_1.z.ZodIssueCode.custom, message: `Storage segment ${segment.segment_id} references unknown node ${segment.node_id}` });
        }
    }
    for (const edge of bundle.graph.edges) {
        if (!canonicalIds.has(edge.source) || !canonicalIds.has(edge.target)) {
            ctx.addIssue({ code: zod_1.z.ZodIssueCode.custom, message: `Graph edge ${edge.id} references unknown nodes` });
        }
    }
});
