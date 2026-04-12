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

        // Make sure a background rect exists BEHIND the dots so drag starts
        // on empty space are captured by the lasso handler.
        if (this.bgRect === undefined || this.bgRect.empty()) {
            this.bgRect = this.svg.insert("rect", ":first-child")
                .attr("class", "lasso-bg")
                .attr("fill", "transparent");
        }
        this.bgRect
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", this.width)
            .attr("height", this.height);

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

        this._attachLasso();
    }

    // Custom D3 v7 compatible lasso. The old d3-lasso library uses d3.event /
    // d3.mouse which were removed in D3 v6+, so we roll our own.
    _attachLasso() {
        // Ensure the lasso path overlay exists
        if (!this.lassoPath) {
            this.lassoPath = this.svg.append("path")
                .attr("class", "lasso-path")
                .attr("fill", "rgba(80, 140, 220, 0.15)")
                .attr("stroke", "rgb(80, 140, 220)")
                .attr("stroke-width", 1.5)
                .attr("stroke-dasharray", "4 3")
                .attr("pointer-events", "none")
                .attr("d", null);
        } else {
            // Lift to the top so it draws above the dots
            this.lassoPath.raise();
        }

        const self = this;
        let polygon = [];

        const drag = d3.drag()
            .container(function () { return this; })  // coordinates relative to <svg>
            .filter((event) => {
                // Only start lasso on the background rect or the svg itself,
                // not on a dot (those have their own click handler).
                const target = event.target;
                return target.tagName !== "circle";
            })
            .on("start", (event) => {
                polygon = [[event.x, event.y]];
                self.lassoPath.attr("d", `M${event.x},${event.y}`);
            })
            .on("drag", (event) => {
                polygon.push([event.x, event.y]);
                self.lassoPath.attr("d", self._polygonPath(polygon));
            })
            .on("end", (event) => {
                if (polygon.length < 3) {
                    // Treated as a click on empty space -- clear selection
                    self.lassoPath.attr("d", null);
                    self.state.clearSelection();
                    polygon = [];
                    return;
                }
                const selectedIds = [];
                self.pointsGroup.selectAll("circle.dot").each(function (d) {
                    const cx = +this.getAttribute("cx");
                    const cy = +this.getAttribute("cy");
                    if (self._pointInPolygon([cx, cy], polygon)) {
                        selectedIds.push(d.id);
                    }
                });
                self.state.setSelection(selectedIds);
                self.lassoPath.attr("d", null);
                polygon = [];
            });

        // Attach to the svg. Calling drag() again replaces the previous handlers.
        this.svg.call(drag);
    }

    _polygonPath(points) {
        if (!points.length) return null;
        return "M" + points.map((p) => `${p[0]},${p[1]}`).join("L") + "Z";
    }

    // Ray-casting point-in-polygon test
    _pointInPolygon(pt, polygon) {
        const [x, y] = pt;
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const [xi, yi] = polygon[i];
            const [xj, yj] = polygon[j];
            const intersect =
                yi > y !== yj > y &&
                x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
            if (intersect) inside = !inside;
        }
        return inside;
    }

    updateSelection(ids) {
        const set = new Set(ids);
        this.pointsGroup.selectAll("circle.dot")
            .classed("selected", (d) => set.has(d.id));
    }
}
