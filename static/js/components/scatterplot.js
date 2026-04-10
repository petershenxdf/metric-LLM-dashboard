// Scatterplot component using D3 + lasso.
// Renders points, supports lasso selection, updates colors on projection changes.

class Scatterplot {
    constructor(containerId, appState) {
        this.container = document.getElementById(containerId);
        this.state = appState;
        this.svg = null;
        this.pointsGroup = null;
        this.lasso = null;
        this.width = 0;
        this.height = 0;

        this._init();
        this._bindState();
    }

    _init() {
        // Clear placeholder
        const ph = document.getElementById("scatterplot-placeholder");
        if (ph) ph.style.display = "none";

        const rect = this.container.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;

        this.svg = d3.select(this.container)
            .append("svg")
            .attr("width", this.width)
            .attr("height", this.height);

        this.pointsGroup = this.svg.append("g").attr("class", "points");

        // Tooltip
        this.tooltip = d3.select(this.container)
            .append("div")
            .attr("class", "scatterplot-tooltip");

        // Resize handling
        window.addEventListener("resize", () => this._resize());
    }

    _bindState() {
        this.state.on("projection_changed", (data) => this.render(data.points));
        this.state.on("selection_changed", (ids) => this.updateSelection(ids));
    }

    _resize() {
        const rect = this.container.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;
        this.svg.attr("width", this.width).attr("height", this.height);
        if (this.state.points && this.state.points.length > 0) {
            this.render(this.state.points);
        }
    }

    render(points) {
        if (!points || points.length === 0) return;

        const padding = 40;
        const xs = points.map((p) => p.x);
        const ys = points.map((p) => p.y);
        const xExtent = [Math.min(...xs), Math.max(...xs)];
        const yExtent = [Math.min(...ys), Math.max(...ys)];

        // Pad extents a bit so points aren't glued to the edge
        const xPad = (xExtent[1] - xExtent[0]) * 0.05 || 1;
        const yPad = (yExtent[1] - yExtent[0]) * 0.05 || 1;

        this.xScale = d3.scaleLinear()
            .domain([xExtent[0] - xPad, xExtent[1] + xPad])
            .range([padding, this.width - padding]);
        this.yScale = d3.scaleLinear()
            .domain([yExtent[0] - yPad, yExtent[1] + yPad])
            .range([this.height - padding, padding]);

        const selectedSet = new Set(this.state.selectedPointIds);

        const dots = this.pointsGroup.selectAll("circle.dot").data(points, (d) => d.id);

        dots.exit().remove();

        const dotsEnter = dots.enter()
            .append("circle")
            .attr("class", "dot");

        const merged = dotsEnter.merge(dots);

        merged
            .attr("cx", (d) => this.xScale(d.x))
            .attr("cy", (d) => this.yScale(d.y))
            .attr("r", 4.5)
            .attr("fill", (d) => colorForCluster(d.cluster))
            .attr("fill-opacity", 0.85)
            .classed("outlier", (d) => d.is_outlier)
            .classed("selected", (d) => selectedSet.has(d.id));

        merged
            .on("mouseover", (event, d) => {
                this.tooltip
                    .classed("visible", true)
                    .style("left", (event.offsetX + 12) + "px")
                    .style("top", (event.offsetY + 12) + "px")
                    .text(`id ${d.id} | cluster ${d.cluster}${d.is_outlier ? " (outlier)" : ""}`);
            })
            .on("mouseout", () => {
                this.tooltip.classed("visible", false);
            })
            .on("click", (event, d) => {
                // Shift-click toggles, plain click selects just this point
                if (event.shiftKey) {
                    const cur = new Set(this.state.selectedPointIds);
                    if (cur.has(d.id)) cur.delete(d.id);
                    else cur.add(d.id);
                    this.state.setSelection(Array.from(cur));
                } else {
                    this.state.setSelection([d.id]);
                }
                event.stopPropagation();
            });

        // Clicking empty space clears selection
        this.svg.on("click", () => this.state.clearSelection());

        this._attachLasso(merged);
    }

    _attachLasso(dotsSelection) {
        if (typeof d3.lasso !== "function") {
            // d3-lasso library not present -- fall back to no lasso
            return;
        }
        if (this.lasso) {
            this.svg.on(".dragstart", null);
        }
        this.lasso = d3.lasso()
            .closePathDistance(75)
            .closePathSelect(true)
            .targetArea(this.svg)
            .items(dotsSelection)
            .on("start", () => {
                dotsSelection.classed("selected", false);
            })
            .on("draw", () => {
                // Optional: visual feedback during drag
            })
            .on("end", () => {
                const selectedIds = this.lasso.selectedItems().data().map((d) => d.id);
                this.state.setSelection(selectedIds);
            });
        this.svg.call(this.lasso);
    }

    updateSelection(ids) {
        const set = new Set(ids);
        this.pointsGroup.selectAll("circle.dot")
            .classed("selected", (d) => set.has(d.id));
    }
}
