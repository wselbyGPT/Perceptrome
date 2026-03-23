from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class RunStartRequest(BaseModel):
    config: dict = Field(default_factory=dict)


class RunArtifactOut(BaseModel):
    id: int
    phase: str | None = None
    path: str
    label: str | None = None
    download_url: str
    created_at: datetime


class ConfigSnapshotOut(BaseModel):
    path: str
    sha256: str
    format: str = "json"


class RunResultOut(BaseModel):
    config_snapshot: ConfigSnapshotOut | None = None


class RunOut(BaseModel):
    run_id: str
    user_id: str
    kind: str
    state: str
    message: str | None = None
    config: dict = Field(default_factory=dict)
    result: RunResultOut | dict | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    artifacts: list[RunArtifactOut] = Field(default_factory=list)


class RunSummaryOut(BaseModel):
    total_runs: int
    state_counts: dict[str, int] = Field(default_factory=dict)
    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    canceled: int = 0
    latest_failed_run_id: str | None = None
    latest_failed_at: datetime | None = None


class RunsBoardOut(BaseModel):
    generated_at: datetime
    runs: list[RunOut] = Field(default_factory=list)


class LineageNodeOut(BaseModel):
    id: str
    kind: str
    label: str
    depth: int = 0
    run_id: str | None = None
    artifact_id: str | None = None
    artifact_type: str | None = None
    run_state: str | None = None
    path: str | None = None
    relation: str | None = None
    hash: str | None = None
    config_snapshot: ConfigSnapshotOut | None = None
    payload: dict = Field(default_factory=dict)


class LineageEdgeOut(BaseModel):
    source: str
    target: str
    relation: str


class RunLineageOut(BaseModel):
    run_id: str
    depth_limit: int
    artifact_type_filter: str | None = None
    run_state_filter: list[str] = Field(default_factory=list)
    nodes: list[LineageNodeOut] = Field(default_factory=list)
    edges: list[LineageEdgeOut] = Field(default_factory=list)


class BioASTCanonicalNodeOut(BaseModel):
    canonical_id: str
    node_type: str
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    start: int | None = None
    end: int | None = None
    strand: str | None = None
    frame: int | None = None
    metadata: dict = Field(default_factory=dict)


class BioASTCanonicalDocumentOut(BaseModel):
    schema: str
    export_version: int | None = None
    schema_version: int | None = None
    accession: str | None = None
    source: str | None = None
    sequence_metadata: dict = Field(default_factory=dict)
    nodes: list[BioASTCanonicalNodeOut] = Field(default_factory=list)
    relationships: list[dict] = Field(default_factory=list)


class BioASTStorageMapTrackOut(BaseModel):
    track_id: str
    track_index: int
    strand: str
    node_type: str
    lane_count: int
    segment_count: int
    segment_range: list[int] = Field(default_factory=list)


class BioASTStorageSegmentOut(BaseModel):
    segment_id: str
    node_id: str
    parent_id: str | None = None
    node_type: str
    strand: str
    track_id: str
    track_index: int
    lane_index: int
    start: int
    end: int
    length: int


class BioASTStorageMapOut(BaseModel):
    schema: str
    accession: str | None = None
    export_version: int | None = None
    sequence_length: int
    topology: dict = Field(default_factory=dict)
    tracks: list[BioASTStorageMapTrackOut] = Field(default_factory=list)
    coordinate_segments: list[BioASTStorageSegmentOut] = Field(default_factory=list)


class BioASTTreeNodeOut(BaseModel):
    id: str
    label: str
    node_type: str
    parent_id: str | None = None
    span: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    coordinates: dict | None = None
    children: list["BioASTTreeNodeOut"] = Field(default_factory=list)


class BioASTTreeOut(BaseModel):
    schema: str
    accession: str | None = None
    export_version: int | None = None
    node_count: int
    sequence_metadata: dict = Field(default_factory=dict)
    roots: list[str] = Field(default_factory=list)
    hierarchy: list[BioASTTreeNodeOut] = Field(default_factory=list)


class BioASTGraphNodeOut(BaseModel):
    id: str
    label: str
    node_type: str
    parent_id: str | None = None
    span: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    coordinates: dict | None = None
    index: int


class BioASTGraphEdgeOut(BaseModel):
    id: str
    source: str
    target: str
    source_index: int
    target_index: int
    relation: str
    edge_kind: str
    relation_type: str
    evidence: str | None = None
    metadata: dict = Field(default_factory=dict)


class BioASTGraphOut(BaseModel):
    schema: str
    accession: str | None = None
    export_version: int | None = None
    node_count: int
    edge_count: int
    hierarchy_edge_count: int
    semantic_edge_count: int
    sequence_metadata: dict = Field(default_factory=dict)
    nodes: list[BioASTGraphNodeOut] = Field(default_factory=list)
    edges: list[BioASTGraphEdgeOut] = Field(default_factory=list)
    hierarchy_edges: list[BioASTGraphEdgeOut] = Field(default_factory=list)
    semantic_edges: list[BioASTGraphEdgeOut] = Field(default_factory=list)


class BioASTBundleResolvedFromOut(BaseModel):
    run_id: str
    artifact_id: int | None = None
    accession: str | None = None
    manifest_path: str | None = None
    base_dir: str


class BioASTVisualizationBundleOut(BaseModel):
    schema: str = 'bio_ast_visualization_bundle_v1'
    accession: str | None = None
    resolved_from: BioASTBundleResolvedFromOut
    canonical: BioASTCanonicalDocumentOut
    storage_map: BioASTStorageMapOut
    tree: BioASTTreeOut
    graph: BioASTGraphOut
    summary: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_cross_payload_ids(self):
        canonical_ids = {node.canonical_id for node in self.canonical.nodes}
        storage_ids = {segment.node_id for segment in self.storage_map.coordinate_segments}
        tree_ids: set[str] = set()

        def visit(nodes: list[BioASTTreeNodeOut]):
            for node in nodes:
                tree_ids.add(node.id)
                visit(node.children)

        visit(self.tree.hierarchy)
        graph_ids = {node.id for node in self.graph.nodes}
        if graph_ids != canonical_ids:
            raise ValueError('Graph node ids must match canonical node ids')
        if not storage_ids.issubset(canonical_ids):
            raise ValueError('Storage map node ids must be a subset of canonical node ids')
        if not tree_ids.issubset(canonical_ids):
            raise ValueError('Tree node ids must be a subset of canonical node ids')
        edge_ids = {(edge.source, edge.target) for edge in self.graph.edges}
        if any(source not in canonical_ids or target not in canonical_ids for source, target in edge_ids):
            raise ValueError('Graph edges must reference canonical node ids')
        return self


BioASTTreeNodeOut.model_rebuild()
