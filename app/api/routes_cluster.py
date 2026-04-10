"""Routes for triggering clustering and getting projection results."""
from flask import Blueprint, request, jsonify, current_app

from app.api.errors import ValidationError, NotFoundError


bp = Blueprint("cluster", __name__, url_prefix="/api/cluster")


@bp.route("/run", methods=["POST"])
def run_clustering():
    """Run a full pipeline iteration: metric update -> SSDBCODI -> MDS."""
    body = request.get_json() or {}
    session_id = body.get("session_id")
    if not session_id:
        raise ValidationError("Missing session_id")

    pipeline = current_app.config["PIPELINE_SERVICE"]
    result = pipeline.run_full_pipeline(session_id)
    return jsonify(result)


@bp.route("/projection/<session_id>", methods=["GET"])
def get_projection(session_id):
    """Return the most recent 2D projection + cluster + outlier info."""
    session_service = current_app.config["SESSION_SERVICE"]
    state = session_service.get(session_id)
    if state is None:
        raise NotFoundError(f"Session {session_id} not found")
    if state.current_projection is None:
        return jsonify({"ready": False, "message": "Run clustering first"})

    return jsonify({
        "ready": True,
        "points": [
            {
                "id": int(i),
                "x": float(state.current_projection[i, 0]),
                "y": float(state.current_projection[i, 1]),
                "cluster": int(state.current_clusters[i]),
                "is_outlier": bool(state.current_outliers[i]),
            }
            for i in range(len(state.current_projection))
        ],
        "n_clusters": int(max(state.current_clusters) + 1) if len(state.current_clusters) > 0 else 0,
        "n_outliers": int(sum(state.current_outliers)),
    })


@bp.route("/summary/<session_id>", methods=["GET"])
def get_cluster_summary(session_id):
    session_service = current_app.config["SESSION_SERVICE"]
    state = session_service.get(session_id)
    if state is None:
        raise NotFoundError(f"Session {session_id} not found")

    if state.current_clusters is None:
        return jsonify({"ready": False})

    cluster_sizes = {}
    for c in state.current_clusters:
        cluster_sizes[int(c)] = cluster_sizes.get(int(c), 0) + 1

    return jsonify({
        "ready": True,
        "n_clusters": len(cluster_sizes),
        "cluster_sizes": cluster_sizes,
        "n_outliers": int(sum(state.current_outliers)) if state.current_outliers is not None else 0,
        "n_constraints": len(state.constraints_history),
    })
