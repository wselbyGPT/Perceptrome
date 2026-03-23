import { useEffect, useMemo, useState } from "react";
import { getBioAstVisualizationBundle } from "./api";
import { createBioAstSelectionStore, type BioASTSelectionState } from "./selection-store";
import type { BioASTTreeNode, BioASTVisualizationBundle } from "./schemas";

function classNames(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

function useSelectionState(runId: string | null) {
  const store = useMemo(() => createBioAstSelectionStore(), [runId]);
  const [state, setState] = useState<BioASTSelectionState>(store.getState());
  useEffect(() => store.subscribe(setState), [store]);
  return { store, state };
}

function TreeBranch({ node, state, store }: { node: BioASTTreeNode; state: BioASTSelectionState; store: ReturnType<typeof createBioAstSelectionStore> }) {
  const active = state.selectedNodeId === node.id || state.hoveredNodeId === node.id;
  return (
    <li>
      <button
        type="button"
        className={classNames("btn btn--secondary btn--sm", active && "bio-ast-selected")}
        onClick={() => store.selectNode(node.id)}
        onMouseEnter={() => store.hoverNode(node.id)}
        onMouseLeave={() => store.hoverNode(null)}
      >
        {node.label}
      </button>
      {node.children.length > 0 ? (
        <ul className="stack-sm" style={{ marginLeft: "1rem", marginTop: "0.5rem" }}>
          {node.children.map((child) => (
            <TreeBranch key={child.id} node={child} state={state} store={store} />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function StorageMapPane({ bundle, state, store }: { bundle: BioASTVisualizationBundle; state: BioASTSelectionState; store: ReturnType<typeof createBioAstSelectionStore> }) {
  const width = 640;
  const trackHeight = 56;
  const laneHeight = 14;
  const totalTracks = Math.max(bundle.storage_map.tracks.length, 1);
  const height = totalTracks * trackHeight + 40;
  const scale = Math.max(bundle.storage_map.sequence_length, 1);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="lineage-graph" role="img" aria-label="Bio-AST storage map">
      {bundle.storage_map.tracks.map((track) => {
        const y = 20 + track.track_index * trackHeight;
        return <text key={track.track_id} x="8" y={y + 10} className="lineage-node-label">{track.node_type} · {track.strand}</text>;
      })}
      {bundle.storage_map.coordinate_segments.map((segment) => {
        const x = 140 + (segment.start / scale) * (width - 160);
        const rectWidth = Math.max(6, ((segment.end - segment.start + 1) / scale) * (width - 160));
        const y = 20 + segment.track_index * trackHeight + 16 + segment.lane_index * laneHeight;
        const active = state.selectedNodeId === segment.node_id || state.hoveredNodeId === segment.node_id;
        return (
          <rect
            key={segment.segment_id}
            x={x}
            y={y}
            width={rectWidth}
            height="10"
            rx="4"
            className={classNames("lineage-node-rect", active && "bio-ast-segment-active")}
            onClick={() => store.selectNode(segment.node_id)}
            onMouseEnter={() => store.hoverNode(segment.node_id)}
            onMouseLeave={() => store.hoverNode(null)}
          />
        );
      })}
    </svg>
  );
}

function GraphPane({ bundle, state, store }: { bundle: BioASTVisualizationBundle; state: BioASTSelectionState; store: ReturnType<typeof createBioAstSelectionStore> }) {
  const width = 640;
  const height = 360;
  const cols = Math.ceil(Math.sqrt(bundle.graph.nodes.length || 1));
  const positions = new Map(bundle.graph.nodes.map((node, index) => [node.id, { x: 60 + (index % cols) * 150, y: 40 + Math.floor(index / cols) * 90 }]));
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="lineage-graph" role="img" aria-label="Bio-AST semantic graph">
      {bundle.graph.edges.map((edge) => {
        const source = positions.get(edge.source);
        const target = positions.get(edge.target);
        if (!source || !target) return null;
        const active = state.selectedEdgeId === edge.id || state.hoveredEdgeId === edge.id || state.selectedNodeId === edge.source || state.selectedNodeId === edge.target;
        return (
          <line
            key={edge.id}
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            stroke={active ? "var(--color-accent, #3b82f6)" : "currentColor"}
            strokeWidth={active ? 2.5 : 1.2}
            onClick={() => store.selectEdge(edge.id)}
            onMouseEnter={() => store.hoverEdge(edge.id)}
            onMouseLeave={() => store.hoverEdge(null)}
          />
        );
      })}
      {bundle.graph.nodes.map((node) => {
        const point = positions.get(node.id)!;
        const active = state.selectedNodeId === node.id || state.hoveredNodeId === node.id;
        return (
          <g key={node.id} transform={`translate(${point.x}, ${point.y})`} onClick={() => store.selectNode(node.id)} onMouseEnter={() => store.hoverNode(node.id)} onMouseLeave={() => store.hoverNode(null)}>
            <circle r="16" className={classNames("lineage-node-rect", active && "bio-ast-segment-active")} />
            <text x="22" y="4" className="lineage-node-label">{node.label.slice(0, 18)}</text>
          </g>
        );
      })}
    </svg>
  );
}

function DetailsPane({ bundle, state }: { bundle: BioASTVisualizationBundle; state: BioASTSelectionState }) {
  const canonicalNode = bundle.canonical.nodes.find((node) => node.canonical_id === state.selectedNodeId || node.canonical_id === state.hoveredNodeId);
  const graphEdge = bundle.graph.edges.find((edge) => edge.id === state.selectedEdgeId || edge.id === state.hoveredEdgeId);
  return <pre className="results-wrap">{JSON.stringify(canonicalNode ?? graphEdge ?? bundle.summary, null, 2)}</pre>;
}

export function BioAstExplorer({ runId, accession, artifactId }: { runId: string | null; accession?: string; artifactId?: number }) {
  const [bundle, setBundle] = useState<BioASTVisualizationBundle | null>(null);
  const [status, setStatus] = useState<string>(runId ? "Loading Bio-AST bundle…" : "Select a run and accession to load Bio-AST.");
  const { store, state } = useSelectionState(runId);

  useEffect(() => {
    if (!runId) {
      setBundle(null);
      setStatus("Select a run and accession to load Bio-AST.");
      return;
    }
    let cancelled = false;
    setStatus("Loading Bio-AST bundle…");
    void getBioAstVisualizationBundle(runId, { accession, artifactId })
      .then((payload) => {
        if (cancelled) return;
        setBundle(payload);
        setStatus(`Loaded ${payload.graph.node_count} nodes from ${payload.resolved_from.base_dir}`);
        store.reset();
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setBundle(null);
        setStatus(`Bio-AST unavailable: ${String(error)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, accession, artifactId, store]);

  return (
    <section className="panel" aria-labelledby="bio-ast-title">
      <div className="panel-header">
        <div>
          <h2 className="panel-title" id="bio-ast-title">Bio-AST Explorer</h2>
          <p className="panel-subtitle">Storage map, hierarchical tree, semantic graph, and synchronized details for the selected run artifact.</p>
        </div>
      </div>
      <p>{status}</p>
      {bundle ? (
        <div className="stack-lg">
          <div className="row" style={{ alignItems: "flex-start" }}>
            <div className="panel" style={{ flex: 1 }}>
              <h3 className="panel-title">Linear storage map</h3>
              <StorageMapPane bundle={bundle} state={state} store={store} />
            </div>
            <div className="panel" style={{ flex: 1 }}>
              <h3 className="panel-title">AST tree</h3>
              <ul className="stack-sm">
                {bundle.tree.hierarchy.map((root) => (
                  <TreeBranch key={root.id} node={root} state={state} store={store} />
                ))}
              </ul>
            </div>
          </div>
          <div className="row" style={{ alignItems: "flex-start" }}>
            <div className="panel" style={{ flex: 1 }}>
              <h3 className="panel-title">Semantic relationship graph</h3>
              <GraphPane bundle={bundle} state={state} store={store} />
            </div>
            <div className="panel" style={{ flex: 1 }}>
              <h3 className="panel-title">Details</h3>
              <DetailsPane bundle={bundle} state={state} />
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
