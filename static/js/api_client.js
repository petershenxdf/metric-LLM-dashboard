// API client: one place for all HTTP calls to the Flask backend.
// Components never call fetch directly -- they go through this class so
// changing routes, adding auth, etc. touches only one file.

class APIClient {
    constructor(baseUrl = "") {
        this.baseUrl = baseUrl;
    }

    async _request(method, path, body = null, isFormData = false) {
        const opts = { method, headers: {} };
        if (body !== null) {
            if (isFormData) {
                opts.body = body;
            } else {
                opts.headers["Content-Type"] = "application/json";
                opts.body = JSON.stringify(body);
            }
        }
        const resp = await fetch(this.baseUrl + path, opts);
        const text = await resp.text();
        let data;
        try {
            data = text ? JSON.parse(text) : {};
        } catch (e) {
            throw new Error(`Invalid JSON response: ${text.slice(0, 200)}`);
        }
        if (!resp.ok) {
            throw new Error(data.error || `HTTP ${resp.status}`);
        }
        return data;
    }

    // Data / session
    async listSamples() {
        return this._request("GET", "/api/data/samples");
    }

    async loadSample(filename) {
        return this._request("POST", "/api/data/load_sample", { filename });
    }

    async uploadDataset(file) {
        const fd = new FormData();
        fd.append("file", file);
        return this._request("POST", "/api/data/upload", fd, true);
    }

    async getDatasetInfo(sessionId) {
        return this._request("GET", `/api/data/info/${sessionId}`);
    }

    // Clustering
    async runClustering(sessionId) {
        return this._request("POST", "/api/cluster/run", { session_id: sessionId });
    }

    async getProjection(sessionId) {
        return this._request("GET", `/api/cluster/projection/${sessionId}`);
    }

    async getClusterSummary(sessionId) {
        return this._request("GET", `/api/cluster/summary/${sessionId}`);
    }

    // Chat
    async sendChatMessage(sessionId, text, selectedIds, selectionGroups) {
        return this._request("POST", "/api/chat/message", {
            session_id: sessionId,
            text: text,
            selected_ids: selectedIds,
            selection_groups: selectionGroups || [],
        });
    }

    // Feedback
    async queueConstraint(sessionId, constraint) {
        // Formerly submitConstraint -- now this only stages a constraint for
        // the next Run Clustering call. The backend no longer re-clusters
        // on submit.
        return this._request("POST", "/api/feedback/submit", {
            session_id: sessionId,
            constraint: constraint,
        });
    }

    async listPending(sessionId) {
        return this._request("GET", `/api/feedback/pending/${sessionId}`);
    }

    async clearPending(sessionId) {
        return this._request("POST", "/api/feedback/pending/clear", {
            session_id: sessionId,
        });
    }

    async undoLast(sessionId) {
        return this._request("POST", "/api/feedback/undo", { session_id: sessionId });
    }

    async listConstraints(sessionId) {
        return this._request("GET", `/api/feedback/list/${sessionId}`);
    }

    // Session
    async resetSession(sessionId) {
        return this._request("POST", "/api/session/reset", { session_id: sessionId });
    }

    // Export
    async getExportSummary(sessionId) {
        return this._request("GET", `/api/export/summary/${sessionId}`);
    }

    exportCSVUrl(sessionId, includeScores = true) {
        // Return the URL that will trigger a browser download when navigated to.
        // We do not fetch this ourselves -- window.location assignment lets
        // the browser handle the Content-Disposition header and show the
        // native save dialog.
        return `${this.baseUrl}/api/export/csv/${sessionId}?include_scores=${includeScores}`;
    }
}
