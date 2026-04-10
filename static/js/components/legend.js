// Legend: renders a small swatch + label for each cluster + outlier category.

class Legend {
    constructor(containerId, appState) {
        this.container = document.getElementById(containerId);
        this.state = appState;
        this.state.on("projection_changed", (data) => this.render(data));
    }

    render(data) {
        if (!data || !data.points) {
            this.container.innerHTML = "";
            return;
        }
        const sizes = {};
        let nOut = 0;
        data.points.forEach((p) => {
            if (p.is_outlier) nOut++;
            else {
                sizes[p.cluster] = (sizes[p.cluster] || 0) + 1;
            }
        });

        const items = [];
        Object.keys(sizes)
            .sort((a, b) => parseInt(a) - parseInt(b))
            .forEach((c) => {
                const color = colorForCluster(parseInt(c));
                items.push(
                    `<div class="legend-item">
                        <span class="legend-swatch" style="background:${color}"></span>
                        Cluster ${c} (${sizes[c]})
                    </div>`
                );
            });
        if (nOut > 0) {
            items.push(
                `<div class="legend-item">
                    <span class="legend-swatch" style="background:${OUTLIER_COLOR};border-style:dashed;border-color:#d32f2f"></span>
                    Outliers (${nOut})
                </div>`
            );
        }

        this.container.innerHTML = items.join("");
    }
}
