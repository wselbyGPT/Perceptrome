export type BioASTSelectionState = {
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  selectedEdgeId: string | null;
  hoveredEdgeId: string | null;
};

export type BioASTSelectionStore = ReturnType<typeof createBioAstSelectionStore>;

const initialState = (): BioASTSelectionState => ({
  selectedNodeId: null,
  hoveredNodeId: null,
  selectedEdgeId: null,
  hoveredEdgeId: null,
});

export function createBioAstSelectionStore() {
  let state = initialState();
  const listeners = new Set<(state: BioASTSelectionState) => void>();

  const emit = () => {
    for (const listener of listeners) listener(state);
  };

  return {
    getState() {
      return state;
    },
    subscribe(listener: (state: BioASTSelectionState) => void) {
      listeners.add(listener);
      listener(state);
      return () => {
        listeners.delete(listener);
      };
    },
    selectNode(nodeId: string | null) {
      state = { ...state, selectedNodeId: nodeId, selectedEdgeId: null };
      emit();
    },
    hoverNode(nodeId: string | null) {
      state = { ...state, hoveredNodeId: nodeId };
      emit();
    },
    selectEdge(edgeId: string | null) {
      state = { ...state, selectedEdgeId: edgeId, selectedNodeId: null };
      emit();
    },
    hoverEdge(edgeId: string | null) {
      state = { ...state, hoveredEdgeId: edgeId };
      emit();
    },
    reset() {
      state = initialState();
      emit();
    },
  };
}
