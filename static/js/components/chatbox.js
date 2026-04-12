// Chatbox component.
// Shows the running chat transcript, the current selection context and any
// staged selection groups, and an input row. On submit it calls the chat
// API. If the response contains a ready-to-apply constraint, it is QUEUED
// via the feedback API -- clustering is not re-run until the user clicks
// "Run clustering" in the control panel.

class Chatbox {
    constructor(containerId, appState, apiClient, onAfterApply) {
        this.container = document.getElementById(containerId);
        this.state = appState;
        this.api = apiClient;
        this.onAfterApply = onAfterApply || (() => {});

        this._build();
        this._bindState();
    }

    _build() {
        this.container.innerHTML = `
            <div class="chat-header">Assistant</div>
            <div class="chat-context" id="chat-context">No points selected.</div>
            <div class="chat-groups" id="chat-groups">
                <div class="chat-groups-row">
                    <span class="chat-groups-label">Constraint groups:</span>
                    <span class="chat-groups-list" id="chat-groups-list">
                        <span class="chat-groups-empty">none staged</span>
                    </span>
                </div>
                <div class="chat-groups-actions">
                    <button id="btn-stage-group" type="button" title="Stage the current selection as the next group (A, B, C...)">Add selection as group</button>
                    <button id="btn-clear-groups" type="button" title="Forget all staged groups">Clear groups</button>
                </div>
                <div class="chat-groups-hint">
                    Stage two groups for cannot-link, or three single points for a triplet (anchor / positive / negative).
                </div>
            </div>
            <div class="chat-messages" id="chat-messages"></div>
            <div class="chat-input-row">
                <textarea id="chat-input" placeholder="Describe what you want (e.g. 'these points are one class')..." rows="1"></textarea>
                <button id="chat-send">Send</button>
            </div>
        `;

        this.contextEl = this.container.querySelector("#chat-context");
        this.groupsListEl = this.container.querySelector("#chat-groups-list");
        this.stageGroupBtn = this.container.querySelector("#btn-stage-group");
        this.clearGroupsBtn = this.container.querySelector("#btn-clear-groups");
        this.messagesEl = this.container.querySelector("#chat-messages");
        this.inputEl = this.container.querySelector("#chat-input");
        this.sendBtn = this.container.querySelector("#chat-send");

        this.sendBtn.addEventListener("click", () => this._handleSend());
        this.inputEl.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                this._handleSend();
            }
        });
        this.inputEl.addEventListener("input", () => {
            this.inputEl.style.height = "auto";
            this.inputEl.style.height = Math.min(this.inputEl.scrollHeight, 120) + "px";
        });

        this.stageGroupBtn.addEventListener("click", () => this._handleStageGroup());
        this.clearGroupsBtn.addEventListener("click", () => {
            this.state.clearSelectionGroups();
        });

        this._renderGroups([]);
        this._updateStageButton([]);
        this.appendSystem("Upload a dataset or load a sample to get started.");
    }

    _bindState() {
        this.state.on("selection_changed", (ids) => {
            this._updateContext(ids);
            this._updateStageButton(ids);
        });
        this.state.on("groups_changed", (groups) => this._renderGroups(groups));
        this.state.on("session_changed", () => {
            this.messagesEl.innerHTML = "";
            this.appendSystem("Session ready. Stage selection groups, describe what you want, then click 'Run clustering'.");
        });
    }

    _updateContext(ids) {
        if (!ids || ids.length === 0) {
            this.contextEl.textContent = "No points selected.";
        } else if (ids.length <= 8) {
            this.contextEl.innerHTML = `Selected: <span class="selection-count">${ids.length}</span> points [${ids.join(", ")}]`;
        } else {
            this.contextEl.innerHTML = `Selected: <span class="selection-count">${ids.length}</span> points`;
        }
    }

    _updateStageButton(ids) {
        this.stageGroupBtn.disabled = !ids || ids.length === 0;
    }

    _renderGroups(groups) {
        if (!groups || groups.length === 0) {
            this.groupsListEl.innerHTML = `<span class="chat-groups-empty">none staged</span>`;
            this.clearGroupsBtn.disabled = true;
            return;
        }
        this.clearGroupsBtn.disabled = false;
        this.groupsListEl.innerHTML = "";
        groups.forEach((g, idx) => {
            const chip = document.createElement("span");
            chip.className = "chat-group-chip";
            chip.title = `Group ${g.label}: ${g.ids.length} point${g.ids.length === 1 ? "" : "s"}`;
            chip.innerHTML = `
                <span class="chat-group-label">${g.label}</span>
                <span class="chat-group-count">${g.ids.length}</span>
                <button type="button" class="chat-group-remove" aria-label="Remove group ${g.label}">&times;</button>
            `;
            chip.querySelector(".chat-group-remove").addEventListener("click", () => {
                this.state.removeGroup(idx);
            });
            this.groupsListEl.appendChild(chip);
        });
    }

    _handleStageGroup() {
        const ok = this.state.stashSelectionAsGroup();
        if (!ok) {
            this.appendError("Select some points first, then click 'Add as group'.");
        }
    }

    appendMessage(role, text) {
        const el = document.createElement("div");
        el.className = `chat-message ${role}`;
        el.textContent = text;
        this.messagesEl.appendChild(el);
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }

    appendSystem(text) {
        this.appendMessage("system", text);
    }

    appendConfirmation(text) {
        const el = document.createElement("div");
        el.className = "chat-message confirmation";
        el.textContent = text;
        this.messagesEl.appendChild(el);
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }

    appendError(text) {
        const el = document.createElement("div");
        el.className = "chat-message error";
        el.textContent = text;
        this.messagesEl.appendChild(el);
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }

    _showLoading() {
        const el = document.createElement("div");
        el.className = "chat-loading";
        el.id = "chat-loading";
        el.textContent = "Thinking...";
        this.messagesEl.appendChild(el);
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }

    _hideLoading() {
        const el = document.getElementById("chat-loading");
        if (el) el.remove();
    }

    async _handleSend() {
        const text = this.inputEl.value.trim();
        if (!text) return;
        if (!this.state.sessionId) {
            this.appendError("Please load a dataset first.");
            return;
        }

        this.appendMessage("user", text);
        this.inputEl.value = "";
        this.inputEl.style.height = "auto";
        this.sendBtn.disabled = true;
        this._showLoading();

        const groupsPayload = this.state.selectionGroups.map((g) => g.ids);

        try {
            const result = await this.api.sendChatMessage(
                this.state.sessionId,
                text,
                this.state.selectedPointIds,
                groupsPayload,
            );
            this._hideLoading();

            if (result.error) {
                this.appendError(result.error);
                return;
            }

            if (result.complete && result.constraint) {
                if (result.confirmation_message) {
                    this.appendConfirmation(result.confirmation_message);
                }
                await this._queueConstraint(result.constraint);
            } else {
                if (result.assistant_message) {
                    this.appendMessage("assistant", result.assistant_message);
                }
            }
        } catch (err) {
            this._hideLoading();
            this.appendError(`Error: ${err.message}`);
        } finally {
            this.sendBtn.disabled = false;
        }
    }

    async _queueConstraint(constraintDict) {
        // Stage the constraint server-side. Do NOT call setProjection --
        // clustering only runs when the user clicks "Run clustering".
        try {
            const result = await this.api.queueConstraint(
                this.state.sessionId,
                constraintDict,
            );
            if (result.error) {
                this.appendError(`Failed to queue: ${result.error}`);
                return;
            }
            if (typeof result.n_pending === "number") {
                this.state.setPendingCount(result.n_pending);
            }
            // Clear staged groups now that they've been consumed by this
            // constraint -- they were the selection state that produced it.
            this.state.clearSelectionGroups();
            this.appendSystem(
                `Queued (${result.n_pending || "?"} pending). Click 'Run clustering' when ready.`,
            );
            this.onAfterApply();
        } catch (err) {
            this.appendError(`Queue error: ${err.message}`);
        }
    }
}
