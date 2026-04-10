// Application entry point. Wires up the components and starts them.

(function () {
    const api = new APIClient();
    const state = new AppState();

    const controlPanel = new ControlPanel("control-bar", state, api);
    const scatterplot = new Scatterplot("scatterplot", state);
    const legend = new Legend("legend", state);
    const chatbox = new Chatbox("chatbox", state, api, () => {});

    // Expose for debugging in the console
    window.app = { api, state, controlPanel, scatterplot, legend, chatbox };
})();
