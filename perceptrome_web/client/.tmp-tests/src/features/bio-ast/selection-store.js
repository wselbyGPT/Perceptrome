"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createBioAstSelectionStore = createBioAstSelectionStore;
const initialState = () => ({
    selectedNodeId: null,
    hoveredNodeId: null,
    selectedEdgeId: null,
    hoveredEdgeId: null,
});
function createBioAstSelectionStore() {
    let state = initialState();
    const listeners = new Set();
    const emit = () => {
        for (const listener of listeners)
            listener(state);
    };
    return {
        getState() {
            return state;
        },
        subscribe(listener) {
            listeners.add(listener);
            listener(state);
            return () => {
                listeners.delete(listener);
            };
        },
        selectNode(nodeId) {
            state = { ...state, selectedNodeId: nodeId, selectedEdgeId: null };
            emit();
        },
        hoverNode(nodeId) {
            state = { ...state, hoveredNodeId: nodeId };
            emit();
        },
        selectEdge(edgeId) {
            state = { ...state, selectedEdgeId: edgeId, selectedNodeId: null };
            emit();
        },
        hoverEdge(edgeId) {
            state = { ...state, hoveredEdgeId: edgeId };
            emit();
        },
        reset() {
            state = initialState();
            emit();
        },
    };
}
