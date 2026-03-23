import { z } from "zod";

export const bioAstCanonicalNodeSchema = z.object({
  canonical_id: z.string(),
  node_type: z.string(),
  parent_id: z.string().nullable().optional(),
  child_ids: z.array(z.string()).default([]),
  start: z.number().int().nullable().optional(),
  end: z.number().int().nullable().optional(),
  strand: z.string().nullable().optional(),
  frame: z.number().int().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).default({}),
});

export const bioAstCanonicalSchema = z.object({
  schema: z.literal("bio_ast_canonical_document_v1"),
  accession: z.string().nullable().optional(),
  export_version: z.number().int().optional(),
  schema_version: z.number().int().optional(),
  sequence_metadata: z.record(z.string(), z.unknown()).default({}),
  nodes: z.array(bioAstCanonicalNodeSchema),
  relationships: z.array(z.record(z.string(), z.unknown())).default([]),
});

export const bioAstStorageTrackSchema = z.object({
  track_id: z.string(),
  track_index: z.number().int(),
  strand: z.string(),
  node_type: z.string(),
  lane_count: z.number().int(),
  segment_count: z.number().int(),
  segment_range: z.array(z.number().int()).length(2),
});

export const bioAstStorageSegmentSchema = z.object({
  segment_id: z.string(),
  node_id: z.string(),
  parent_id: z.string().nullable().optional(),
  node_type: z.string(),
  strand: z.string(),
  track_id: z.string(),
  track_index: z.number().int(),
  lane_index: z.number().int(),
  start: z.number().int(),
  end: z.number().int(),
  length: z.number().int(),
});

export const bioAstStorageMapSchema = z.object({
  schema: z.literal("bio_ast_storage_map_v1"),
  accession: z.string().nullable().optional(),
  export_version: z.number().int().optional(),
  sequence_length: z.number().int(),
  topology: z.record(z.string(), z.unknown()).default({}),
  tracks: z.array(bioAstStorageTrackSchema),
  coordinate_segments: z.array(bioAstStorageSegmentSchema),
});

export type BioASTTreeNode = {
  id: string;
  label: string;
  node_type: string;
  parent_id?: string | null;
  span: Record<string, unknown>;
  metadata: Record<string, unknown>;
  coordinates?: Record<string, unknown> | null;
  children: BioASTTreeNode[];
};

export const bioAstTreeNodeSchema: z.ZodType<BioASTTreeNode> = z.lazy(() =>
  z.object({
    id: z.string(),
    label: z.string(),
    node_type: z.string(),
    parent_id: z.string().nullable().optional(),
    span: z.record(z.string(), z.unknown()).default({}),
    metadata: z.record(z.string(), z.unknown()).default({}),
    coordinates: z.record(z.string(), z.unknown()).nullable().optional(),
    children: z.array(bioAstTreeNodeSchema).default([]),
  }),
);

export const bioAstTreeSchema = z.object({
  schema: z.literal("bio_ast_tree_v1"),
  accession: z.string().nullable().optional(),
  export_version: z.number().int().optional(),
  node_count: z.number().int(),
  sequence_metadata: z.record(z.string(), z.unknown()).default({}),
  roots: z.array(z.string()),
  hierarchy: z.array(bioAstTreeNodeSchema),
});

export const bioAstGraphNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  node_type: z.string(),
  parent_id: z.string().nullable().optional(),
  span: z.record(z.string(), z.unknown()).default({}),
  metadata: z.record(z.string(), z.unknown()).default({}),
  coordinates: z.record(z.string(), z.unknown()).nullable().optional(),
  index: z.number().int(),
});

export const bioAstGraphEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  source_index: z.number().int(),
  target_index: z.number().int(),
  relation: z.string(),
  edge_kind: z.string(),
  relation_type: z.string(),
  evidence: z.string().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).default({}),
});

export const bioAstGraphSchema = z.object({
  schema: z.literal("bio_ast_graph_v1"),
  accession: z.string().nullable().optional(),
  export_version: z.number().int().optional(),
  node_count: z.number().int(),
  edge_count: z.number().int(),
  hierarchy_edge_count: z.number().int(),
  semantic_edge_count: z.number().int(),
  sequence_metadata: z.record(z.string(), z.unknown()).default({}),
  nodes: z.array(bioAstGraphNodeSchema),
  edges: z.array(bioAstGraphEdgeSchema),
  hierarchy_edges: z.array(bioAstGraphEdgeSchema),
  semantic_edges: z.array(bioAstGraphEdgeSchema),
});

export const bioAstVisualizationBundleSchema = z
  .object({
    schema: z.literal("bio_ast_visualization_bundle_v1"),
    accession: z.string().nullable().optional(),
    resolved_from: z.object({
      run_id: z.string(),
      artifact_id: z.number().int().nullable().optional(),
      accession: z.string().nullable().optional(),
      manifest_path: z.string().nullable().optional(),
      base_dir: z.string(),
    }),
    canonical: bioAstCanonicalSchema,
    storage_map: bioAstStorageMapSchema,
    tree: bioAstTreeSchema,
    graph: bioAstGraphSchema,
    summary: z.record(z.string(), z.unknown()).default({}),
  })
  .superRefine((bundle, ctx) => {
    const canonicalIds = new Set(bundle.canonical.nodes.map((node) => node.canonical_id));
    const treeIds = new Set<string>();
    const visit = (nodes: BioASTTreeNode[]) => {
      for (const node of nodes) {
        treeIds.add(node.id);
        visit(node.children);
      }
    };
    visit(bundle.tree.hierarchy);

    for (const nodeId of treeIds) {
      if (!canonicalIds.has(nodeId)) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Tree node ${nodeId} is missing from canonical payload` });
      }
    }
    for (const segment of bundle.storage_map.coordinate_segments) {
      if (!canonicalIds.has(segment.node_id)) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Storage segment ${segment.segment_id} references unknown node ${segment.node_id}` });
      }
    }
    for (const edge of bundle.graph.edges) {
      if (!canonicalIds.has(edge.source) || !canonicalIds.has(edge.target)) {
        ctx.addIssue({ code: z.ZodIssueCode.custom, message: `Graph edge ${edge.id} references unknown nodes` });
      }
    }
  });

export type BioASTVisualizationBundle = z.infer<typeof bioAstVisualizationBundleSchema>;
