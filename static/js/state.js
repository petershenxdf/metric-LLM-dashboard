// Global app state with a minimal pub/sub so components can subscribe to
// changes without knowing about each other.

class AppState {
    constructor() {
        this.sessionId = null;
        this.datasetInfo = null;       // { n_points, n_features, feature_names, ... }
        this.selectedPointIds = [];    // array of ints: the "live" selection
        // Staged selection groups: each entry is { label, ids }. Used for
        // constraints that need multiple sets of points (cannot-link needs
        // group_a + group_b, triplet needs anchor + positive + negative).
        // The chatbox sends this list alongside the live selection.
        this.selectionGroups = [];
        this.points = [];              // array of { id, x, y, cluster, is_outlier }
        this.nClusters = 0;
        this.nOutliers = 0;
        this.nConstraints = 0;
        this.nPending = 0;             // queued-but-not-yet-applied constraints

        this._listeners = {};           // event name -> [callback, ...]
    }

    on(event, callback) {
        if (!this._listeners[event]) this._listeners[event] = [];
        this._listeners[event].push(callback);
    }

    emit(event, payload) {
        (this._listeners[event] || []).forEach((cb) => cb(payload));
    }

    setSession(sessionId, datasetInfo) {
        this.sessionId = sessionId;
        this.datasetInfo = datasetInfo;
        this.selectedPointIds = [];
        this.selectionGroups = [];
        this.points = [];
        this.nClusters = 0;
        this.nOutliers = 0;
        this.nConstraints = 0;
        this.nPending = 0;
        this.emit("session_changed", { sessionId, datasetInfo });
        this.emit("groups_changed", this.selectionGroups);
        this.emit("pending_changed", this.nPending);
    }

    setProjection(data) {
        this.points = data.points || [];
        this.nClusters = data.n_clusters || 0;
        this.nOutliers = data.n_outliers || 0;
        this.nConstraints = data.n_constraints || this.nConstraints;
        if (typeof data.n_pending === "number") {
            this.nPending = data.n_pending;
            this.emit("pending_changed", this.nPending);
        }
        this.emit("projection_changed", data);
    }

    setSelection(ids) {
        this.selectedPointIds = Array.from(new Set(ids)).sort((a, b) => a - b);
        this.emit("selection_changed", this.selectedPointIds);
    }

    clearSelection() {
        this.setSelection([]);
    }

    // Selection group management ---------------------------------------

    stashSelectionAsGroup() {
        // Snapshot the current live selection into a new group slot, then
        // clear the live selection so the user can pick the next set.
        if (!this.selectedPointIds.length) return false;
        const label = this._nextGroupLabel();
        this.selectionGroups.push({
            label,
            ids: this.selectedPointIds.slice(),
        });
        this.emit("groups_changed", this.selectionGroups);
        this.setSelection([]);
        return true;
    }

    removeGroup(index) {
        if (index < 0 || index >= this.selectionGroups.length) return;
        this.selectionGroups.splice(index, 1);
        // Relabel everything so labels stay contiguous (A, B, C, ...)
        this.selectionGroups.forEach((g, i) => {
            g.label = this._groupLabelForIndex(i);
        });
        this.emit("groups_changed", this.selectionGroups);
    }

    clearSelectionGroups() {
        if (!this.selectionGroups.length) return;
        this.selectionGroups = [];
        this.emit("groups_changed", this.selectionGroups);
    }

    setPendingCount(n) {
        this.nPending = n;
        this.emit("pending_changed", this.nPending);
    }

    _nextGroupLabel() {
        return this._groupLabelForIndex(this.selectionGroups.length);
    }

    _groupLabelForIndex(i) {
        // A, B, C, ..., Z, then AA, AB, ...
        const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        if (i < 26) return letters[i];
        return letters[Math.floor(i / 26) - 1] + letters[i % 26];
    }
}
