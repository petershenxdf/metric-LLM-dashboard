// Global app state with a minimal pub/sub so components can subscribe to
// changes without knowing about each other.

class AppState {
    constructor() {
        this.sessionId = null;
        this.datasetInfo = null;       // { n_points, n_features, feature_names, ... }
        this.selectedPointIds = [];    // array of ints
        this.points = [];              // array of { id, x, y, cluster, is_outlier }
        this.nClusters = 0;
        this.nOutliers = 0;
        this.nConstraints = 0;

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
        this.points = [];
        this.nClusters = 0;
        this.nOutliers = 0;
        this.nConstraints = 0;
        this.emit("session_changed", { sessionId, datasetInfo });
    }

    setProjection(data) {
        this.points = data.points || [];
        this.nClusters = data.n_clusters || 0;
        this.nOutliers = data.n_outliers || 0;
        this.nConstraints = data.n_constraints || this.nConstraints;
        this.emit("projection_changed", data);
    }

    setSelection(ids) {
        this.selectedPointIds = Array.from(new Set(ids)).sort((a, b) => a - b);
        this.emit("selection_changed", this.selectedPointIds);
    }

    clearSelection() {
        this.setSelection([]);
    }
}
