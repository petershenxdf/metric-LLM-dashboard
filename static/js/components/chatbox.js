// Chatbox component.
// Shows the running chat transcript, the current selection context, and an
// input row. On submit it calls the chat API. If the response contains a
// ready-to-apply constraint, it is auto-submitted via the feedback API.

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
            <div class="chat-messages" id="chat-messages"></div>
            <div class="chat-input-row">
                <textarea id="chat-input" placeholder="Describe what you want (e.g. 'these points are one class')..." rows="1"></textarea>
                <button id="chat-send">Send</button>
            </div>
        `;

        this.contextEl = this.container.querySelector("#chat-context");
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

        this.appendSystem("Upload a dataset or load a sample to get started.");
    }

    _bindState() {
        this.state.on("selection_changed", (ids) => this._updateContext(ids));
        this.state.on("session_changed", () => {
            this.messagesEl.innerHTML = "";
            this.appendSystem("Session ready. Run clustering, then give me instructions.");
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

        try {
            const result = await this.api.sendChatMessage(
                this.state.sessionId,
                text,
                this.state.selectedPointIds
            );
            this._hideLoading();

            if (result.error) {
                this.appendError(result.error);
                return;
            }

            if (result.complete && result.constraint) {
                // Got a ready constraint -- show confirmation and apply
                if (result.confirmation_message) {
                    this.appendConfirmation(result.confirmation_message);
                }
                await this._applyConstraint(result.constraint);
            } else {
                // Not complete -- show follow-up
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

    async _applyConstraint(constraintDict) {
        try {
            const result = await this.api.submitConstraint(
                this.state.sessionId,
                constraintDict
            );
            if (result.error) {
                this.appendError(`Failed to apply: ${result.error}`);
                return;
            }
            this.state.setProjection(result);
            this.appendSystem("Updated clustering applied.");
            this.onAfterApply();
        } catch (err) {
            this.appendError(`Apply error: ${err.message}`);
        }
    }
}
