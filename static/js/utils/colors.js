// Cluster color palette. d3's schemeTableau10 is colorblind-friendly and
// matches the aesthetic of the dashboard.

const CLUSTER_PALETTE = [
    "#4e79a7", "#f28e2b", "#a357e1", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
];

const OUTLIER_COLOR = "#dd0808";

function colorForCluster(clusterId) {
    if (clusterId < 0) return OUTLIER_COLOR;
    return CLUSTER_PALETTE[clusterId % CLUSTER_PALETTE.length];
}
