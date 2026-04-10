// Control panel: upload, sample selector, run clustering, undo, reset.

class ControlPanel {
    constructor(containerId, appState, apiClient) {
        this.container = document.getElementById(containerId);
        this.state = appState;
        this.api = apiClient;
        this._build();
        this._loadSamples();
    }

    _build() {
        this.container.innerHTML = `
            <input type="file" id="file-input" accept=".csv,.tsv,.txt" style="display:none">
            <button id="btn-upload">Upload CSV</button>
            <select id="sample-select">
                <option value="">-- Load sample --</option>
            </select>
            <span class="separator"></span>
            <button id="btn-run" class="primary" disabled>Run clustering</button>
            <button id="btn-undo" disabled>Undo</button>
            <button id="btn-reset" disabled>Reset</button>
            <span class="separator"></span>
            <button id="btn-export" disabled title="Download the labeled dataset as CSV">Export CSV</button>
            <span class="info-text" id="info-text">No dataset loaded</span>
        `;

        this.fileInput = this.container.querySelector("#file-input");
        this.uploadBtn = this.container.querySelector("#btn-upload");
        this.sampleSelect = this.container.querySelector("#sample-select");
        this.runBtn = this.container.querySelector("#btn-run");
        this.undoBtn = this.container.querySelector("#btn-undo");
        this.resetBtn = this.container.querySelector("#btn-reset");
        this.exportBtn = this.container.querySelector("#btn-export");
        this.infoText = this.container.querySelector("#info-text");

        this.uploadBtn.addEventListener("click", () => this.fileInput.click());
        this.fileInput.addEventListener("change", (e) => this._handleUpload(e));
        this.sampleSelect.addEventListener("change", (e) => this._handleLoadSample(e));
        this.runBtn.addEventListener("click", () => this._handleRun());
        this.undoBtn.addEventListener("click", () => this._handleUndo());
        this.resetBtn.addEventListener("click", () => this._handleReset());
        this.exportBtn.addEventListener("click", () => this._handleExport());

        this.state.on("session_changed", () => this._onSessionChanged());
        this.state.on("projection_changed", (data) => this._onProjectionChanged(data));
    }

    async _loadSamples() {
        try {
            const data = await this.api.listSamples();
            (data.samples || []).forEach((s) => {
                const opt = document.createElement("option");
                opt.value = s.filename;
                opt.textContent = s.name;
                this.sampleSelect.appendChild(opt);
            });
        } catch (err) {
            console.error("Failed to load sample list:", err);
        }
    }

    async _handleUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        this._setStatus("working", "Uploading...");
        try {
            const result = await this.api.uploadDataset(file);
            this.state.setSession(result.session_id, result);
            this._setStatus("ready", "Ready");
            if (result.warnings && result.warnings.length) {
                console.warn("Upload warnings:", result.warnings);
            }
        } catch (err) {
            this._setStatus("error", "Upload failed");
            alert(`Upload failed: ${err.message}`);
        }
        this.fileInput.value = "";
    }

    async _handleLoadSample(e) {
        const filename = e.target.value;
        if (!filename) return;
        this._setStatus("working", "Loading sample...");
        try {
            const result = await this.api.loadSample(filename);
            this.state.setSession(result.session_id, result);
            this._setStatus("ready", "Ready");
        } catch (err) {
            this._setStatus("error", "Load failed");
            alert(`Load failed: ${err.message}`);
        }
        this.sampleSelect.value = "";
    }

    async _handleRun() {
        if (!this.state.sessionId) return;
        this._setStatus("working", "Clustering...");
        this.runBtn.disabled = true;
        try {
            const result = await this.api.runClustering(this.state.sessionId);
            this.state.setProjection(result);
            this._setStatus("ready", "Ready");
        } catch (err) {
            this._setStatus("error", "Clustering failed");
            alert(`Clustering failed: ${err.message}`);
        } finally {
            this.runBtn.disabled = false;
        }
    }

    async _handleUndo() {
        if (!this.state.sessionId) return;
        this._setStatus("working", "Undoing...");
        try {
            const result = await this.api.undoLast(this.state.sessionId);
            if (result.error) {
                alert(result.error);
            } else {
                this.state.setProjection(result);
            }
            this._setStatus("ready", "Ready");
        } catch (err) {
            this._setStatus("error", "Undo failed");
            alert(`Undo failed: ${err.message}`);
        }
    }

    async _handleReset() {
        if (!this.state.sessionId) return;
        if (!confirm("Reset all constraints and clustering for this session?")) return;
        try {
            await this.api.resetSession(this.state.sessionId);
            await this._handleRun();
        } catch (err) {
            alert(`Reset failed: ${err.message}`);
        }
    }

    _handleExport() {
        if (!this.state.sessionId) return;
        // Navigate to the export URL -- the browser will handle the download
        // via the Content-Disposition header from the backend.
        const url = this.api.exportCSVUrl(this.state.sessionId, true);
        window.location.href = url;
    }

    _onSessionChanged() {
        if (this.state.sessionId) {
            this.runBtn.disabled = false;
            this.resetBtn.disabled = false;
            const di = this.state.datasetInfo;
            this.infoText.textContent = `${di.n_points} points, ${di.n_features} features`;
        } else {
            this.runBtn.disabled = true;
            this.resetBtn.disabled = true;
            this.undoBtn.disabled = true;
            this.exportBtn.disabled = true;
            this.infoText.textContent = "No dataset loaded";
        }
    }

    _onProjectionChanged(data) {
        if (data && data.n_constraints > 0) {
            this.undoBtn.disabled = false;
        }
        // Enable export as soon as any clustering result is available
        if (data && data.ready) {
            this.exportBtn.disabled = false;
        }
        if (data) {
            const di = this.state.datasetInfo || {};
            this.infoText.textContent =
                `${di.n_points || "?"} points | ${data.n_clusters || 0} clusters | ${data.n_outliers || 0} outliers`;
        }
    }

    _setStatus(cls, text) {
        const dot = document.getElementById("status-indicator");
        const txt = document.getElementById("status-text");
        if (dot) {
            dot.classList.remove("ready", "working", "error");
            dot.classList.add(cls);
        }
        if (txt) txt.textContent = text;
    }
}
