# SSDBCODI Interactive Dashboard — Complete Technical Documentation

> A semi-supervised clustering and outlier-detection system guided by natural-language feedback.
> Code reviewed from the latest committed state (April 2026).

---

## Table of Contents

1. [What the Project Does](#1-what-the-project-does)
2. [High-Level Architecture](#2-high-level-architecture)
3. [End-to-End User Workflow](#3-end-to-end-user-workflow)
4. [Detailed System Flowchart](#4-detailed-system-flowchart)
5. [Layer-by-Layer Breakdown](#5-layer-by-layer-breakdown)
   - 5.1 [Presentation Layer (Frontend)](#51-presentation-layer-frontend)
   - 5.2 [API Layer (Flask Blueprints)](#52-api-layer-flask-blueprints)
   - 5.3 [Service Layer (Orchestration)](#53-service-layer-orchestration)
   - 5.4 [Domain Layer (Core Algorithms)](#54-domain-layer-core-algorithms)
   - 5.5 [Infrastructure Layer (External Boundaries)](#55-infrastructure-layer-external-boundaries)
6. [The SSDBCODI Algorithm Explained](#6-the-ssdbcodi-algorithm-explained)
7. [The Chat / Intent Pipeline](#7-the-chat--intent-pipeline)
8. [The Constraint System](#8-the-constraint-system)
9. [The Metric Learning System](#9-the-metric-learning-system)
10. [Session State and Undo](#10-session-state-and-undo)
11. [Data Flow: From User Message to Updated Scatterplot](#11-data-flow-from-user-message-to-updated-scatterplot)
12. [File Map](#12-file-map)

---

## 1. What the Project Does

SSDBCODI Dashboard is an **interactive machine-learning workbench** that lets a non-technical user iteratively refine a clustering result by chatting with an AI assistant instead of manually tweaking algorithm hyperparameters or writing code.

The core research idea combines two things:

- **SSDBCODI** (Semi-Supervised Density-Based Clustering with Outlier Detection): a density-based algorithm that handles non-convex clusters and outliers jointly. It accepts a small set of labeled points (normal and outlier) to guide its expansion, then uses a trained classifier to assign the rest.
- **Interactive Metric Learning**: a Mahalanobis distance matrix **M** that is updated after every user instruction. The algorithm re-runs with the updated M so the geometric notion of "closeness" changes in response to what the user said.

The net effect: the user loads a CSV, sees a 2D scatterplot, selects some points, types something like *"these are all the same class"* or *"ignore the color column"*, and the scatterplot immediately rearranges to reflect that guidance — all without writing a single line of code.

---

## 2. High-Level Architecture

The codebase is split into five strictly layered tiers. Dependency arrows always point **downward** — upper layers import lower layers, never the reverse. The domain layer has no I/O at all and can be unit-tested in complete isolation.

```
┌─────────────────────────────────────────────────────────────┐
│  PRESENTATION   static/ + templates/                        │
│  D3.js scatterplot · lasso selection · chatbox · pub/sub    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (JSON)
┌────────────────────────▼────────────────────────────────────┐
│  API LAYER      app/api/                                     │
│  Thin Flask blueprints — routes_data, _chat, _feedback,     │
│  _cluster, _session, _debug, _export                        │
└────────────────────────┬────────────────────────────────────┘
                         │ function calls
┌────────────────────────▼────────────────────────────────────┐
│  SERVICE LAYER  app/services/                               │
│  chat_service · feedback_service · pipeline_service         │
│  session_service                                            │
└────────────────────────┬────────────────────────────────────┘
                         │ function calls
┌────────────────────────▼────────────────────────────────────┐
│  DOMAIN LAYER   app/domain/  (zero I/O)                     │
│  clustering/  · metric_learning/  · constraints/            │
│  projection/  · intent/                                     │
└────────────────────────┬────────────────────────────────────┘
                         │ abstract interfaces
┌────────────────────────▼────────────────────────────────────┐
│  INFRASTRUCTURE  app/infrastructure/                        │
│  llm/ · data/ · storage/ · debug/                           │
│  (everything external — swappable via .env + factory)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End User Workflow

```
User loads CSV
      │
      ▼
Backend normalises data, creates session
      │
      ▼
Default SSDBCODI clustering runs (identity M, cold-start)
      │
      ▼
Frontend renders 2D scatterplot (MDS projection)
      │
 ┌────┴────────────────────────────────────────────────────┐
 │          INTERACTIVE LOOP (repeats until done)          │
 │                                                         │
 │  1. User selects points with lasso / shift-click        │
 │  2. User types instruction in chatbox                   │
 │  3. Chat service classifies intent:                     │
 │     a. Regex rule engine (fast path)                    │
 │     b. LLM fallback (Ollama / OpenAI-compat)            │
 │  4. If intent is incomplete → follow-up question        │
 │  5. If intent is complete → structured constraint JSON  │
 │  6. Constraint auto-submitted to feedback API           │
 │  7. Constraint routed to LABEL channel and/or METRIC    │
 │     channel                                             │
 │  8. M updated (if metric channel)                       │
 │  9. DN/DO updated (if label channel)                    │
 │  10. SSDBCODI re-runs with new M + new labels           │
 │  11. MDS re-projects with new M                         │
 │  12. Scatterplot refreshes — user sees effect           │
 └────────────────────────────────────────────────────────┘
      │
      ▼
User clicks "Export CSV"
      │
      ▼
Original column values + cluster_label + is_outlier +
rscore, lscore, simscore, tscore downloaded
```

---

## 4. Detailed System Flowchart

The diagram below traces a single full iteration from the moment the user presses **Send** in the chatbox to the moment the scatterplot refreshes.

```mermaid
flowchart TD
    A([User types message\nin chatbox]) --> B[Chatbox._handleSend]
    B --> C[POST /api/chat/message\n{session_id, text, selected_ids}]
    C --> D[chat_service.process_message]

    D --> E{Rule classifier\nfinds a match?}
    E -- Yes --> F[Build constraint dict\nfrom slots + selected_ids]
    E -- No --> G[LLM classifier\nbuild prompt with:\n- selected_ids\n- cluster_summary\n- n_points\n- few-shot examples]
    G --> H[LLMClient.chat\nOllama / OpenAI]
    H --> I[Parse JSON response]

    F --> J{Constraint\ncomplete & valid?}
    I --> J

    J -- No --> K[Return follow-up\nquestion to frontend]
    K --> A

    J -- Yes --> L[Return constraint dict\n+ confirmation_message]
    L --> M[Chatbox auto-submits\nPOST /api/feedback/submit]

    M --> N[feedback_service.apply_constraint]
    N --> N1[state.snapshot  ← undo point saved]
    N1 --> O[router.route_constraint]

    O --> P{Which channel?}

    P -- LABEL or BOTH --> Q[_apply_to_label_channel]
    Q --> Q1{Constraint type?}
    Q1 -- must_link --> Q2[Set shared cluster_id\nin state.DN for all points]
    Q1 -- outlier_label --> Q3[Add/remove from state.DO]
    Q1 -- cluster_merge --> Q4[Rewrite DN labels\nto target cluster]
    Q1 -- reassign --> Q5[Move points to\ntarget_cluster_id in DN]
    Q1 -- cluster_count --> Q6[Stored in history;\nalgorithm uses as hint]

    P -- METRIC or BOTH --> R[_apply_to_metric_channel]
    R --> S[CompositeMetricLearner.update]
    S --> S1{Constraint type?}
    S1 -- must_link --> S2[ITML: add similar pairs\nfrom all-pairs of point_ids]
    S1 -- cannot_link --> S3[ITML: add dissimilar pairs\nfrom cross-product of groups]
    S1 -- triplet --> S4[TripletLearner SGD step\nanchor closer to pos than neg]
    S1 -- feature_hint --> S5[Edit diagonal of M:\nscale / zero out feature]
    S2 --> S6[ITMLLearner.update\nproject M to PSD cone]
    S3 --> S6
    S4 --> S7[TripletLearner.update\nproject M to PSD cone]
    S5 --> S8[Re-symmetrize,\nPSD project M]
    S6 --> T[Sync M into both sub-learners]
    S7 --> T
    S8 --> T
    T --> U[Return updated M]

    Q2 & Q3 & Q4 & Q5 & Q6 --> V[session_service.save state]
    U --> V

    V --> W[pipeline_service.apply_constraint_and_recluster]
    W --> X[Store M on state\nAppend constraint to history]
    X --> Y[pipeline_service.run_full_pipeline]

    Y --> Z[MahalanobisDistance from M]
    Z --> AA[SSDBCODI.fit\nX, DN, DO, distance_func]

    AA --> AA1[ssdbscan:\nprecompute dist + rDist matrices\nExpand from each labeled point\nusing Prim-like traversal]
    AA1 --> AA2[Compute scores:\nrScore, lScore, simScore, tScore]
    AA2 --> AA3[Build RN reliable normals\nBuild RO reliable outliers\ntop-k by tScore]
    AA3 --> AA4[Train RandomForest\non RN + RO\nweighted by scores]
    AA4 --> AA5[Predict every point\nOverride with explicit\nDN / DO labels]

    AA5 --> BB[MDSProjector.project\nprecompute M-dist matrix\nrun sklearn MDS]
    BB --> CC[Update session:\ncurrent_clusters\ncurrent_outliers\ncurrent_projection\ncurrent_scores]
    CC --> DD[_build_response:\narray of {id,x,y,cluster,is_outlier}]
    DD --> EE[JSON response to frontend]
    EE --> FF[AppState.setProjection\nemit projection_changed]
    FF --> GG[Scatterplot.render\nD3 re-draws all circles\ncolored by cluster]
    GG --> HH([User sees updated\nscatterplot])
```

---

## 5. Layer-by-Layer Breakdown

### 5.1 Presentation Layer (Frontend)

**Files:** `static/js/`, `static/css/`, `templates/index.html`

The frontend is plain vanilla JavaScript with D3 v7. There is no build step. Components communicate through a shared `AppState` pub/sub bus rather than calling each other directly.

#### `AppState` (`state.js`)

The single source of truth on the client side. Holds:

| Field | Type | Description |
|---|---|---|
| `sessionId` | string | UUID from the backend |
| `datasetInfo` | object | n_points, n_features, feature_names |
| `selectedPointIds` | int[] | Currently lasso/click-selected point indices |
| `points` | object[] | `{id, x, y, cluster, is_outlier}` per data point |
| `nClusters` | int | Number of clusters in current result |
| `nOutliers` | int | Number of outliers in current result |
| `nConstraints` | int | Total constraints submitted this session |

Events emitted: `session_changed`, `projection_changed`, `selection_changed`.

#### `ControlPanel` (`components/control_panel.js`)

The top bar. Handles:
- Upload CSV → `POST /api/data/upload`
- Load sample → `POST /api/data/load_sample`
- Run clustering → `POST /api/cluster/run` → `state.setProjection()`
- Undo → `POST /api/feedback/undo`
- Reset → `POST /api/session/reset/{session_id}`
- Export CSV → `GET /api/export/csv/{session_id}`

#### `Scatterplot` (`components/scatterplot.js`)

D3-powered SVG canvas. On every `projection_changed` event it re-renders all data points as circles:
- Color determined by `colorForCluster(cluster_id)` — a stable categorical palette
- Outliers get CSS class `outlier` → dashed border via CSS
- Selected points get CSS class `selected` → orange ring
- Lasso selection via d3-lasso plugin → calls `state.setSelection(ids)`
- Shift-click toggles individual points in/out of selection

#### `Chatbox` (`components/chatbox.js`)

The interactive assistant panel:
1. Shows running conversation transcript
2. Displays current selection count in a context bar at the top
3. On **Send**: calls `POST /api/chat/message` with `{session_id, text, selected_ids}`
4. If the response has `complete: true`, **automatically** calls `POST /api/feedback/submit` with the constraint — no extra click needed
5. If `complete: false`, shows the follow-up question and waits for the user's reply

#### `Legend` (`components/legend.js`)

Renders the cluster colour legend, updating whenever `projection_changed` fires.

---

### 5.2 API Layer (Flask Blueprints)

**Files:** `app/api/routes_*.py`

Each blueprint is thin — it validates the incoming request, pulls the right service object from `current_app.config`, calls it, and returns `jsonify(result)`. No business logic lives here.

| Blueprint | Prefix | Key endpoints |
|---|---|---|
| `routes_data` | `/api/data` | `POST /upload`, `POST /load_sample`, `GET /samples`, `GET /info/<id>` |
| `routes_cluster` | `/api/cluster` | `POST /run` |
| `routes_chat` | `/api/chat` | `POST /message`, `GET /history/<id>` |
| `routes_feedback` | `/api/feedback` | `POST /submit`, `POST /undo`, `GET /list/<id>` |
| `routes_session` | `/api/session` | `GET /list`, `DELETE /reset/<id>` |
| `routes_debug` | `/api/debug` | Various inspection endpoints |
| `routes_export` | `/api/export` | `GET /csv/<id>` |

**Error handling** (`errors.py`): `ValidationError` → 400, `NotFoundError` → 404, all others → 500, always JSON body.

---

### 5.3 Service Layer (Orchestration)

**Files:** `app/services/`

Services know *what* to do and in *what order*, but not *how* the domain algorithms work internally.

#### `SessionService`

Wraps the storage backend (in-memory dict or pickle file). Provides `get(session_id)`, `save(state)`, `rollback(session_id)`. The storage backend is swappable via `.env` without touching service code.

#### `ChatService`

Processes one user message. Flow:

```
user_text + selected_ids
        │
        ▼
RuleClassifier.classify(text)
        │
   match? ──Yes──► build constraint dict from slots
        │                        │
       No                        │
        ▼                        │
LLMIntentClassifier.classify    │
  (prompt + few-shot examples)  │
        │                        │
        ▼                        ▼
   complete=False?         complete=True → validate() → return constraint
        │
        ▼
   return follow-up question
```

#### `FeedbackService`

Applies a structured constraint to the session, then triggers a re-cluster. Per-session `CompositeMetricLearner` instances are cached here so they accumulate M updates across multiple constraints.

Routing logic (via `domain/constraints/router.py`):

| Constraint | Label channel | Metric channel |
|---|:---:|:---:|
| `must_link` | ✓ (assign cluster) | ✓ (ITML similar pairs) |
| `cannot_link` | — | ✓ (ITML dissimilar pairs) |
| `triplet` | — | ✓ (triplet SGD) |
| `cluster_count` | ✓ (stored as hint) | — |
| `outlier_label` | ✓ (update DO) | — |
| `feature_hint` | — | ✓ (edit M diagonal) |
| `cluster_merge` | ✓ (rewrite DN) | — |
| `reassign` | ✓ (update DN) | — |

#### `PipelineService`

Runs one full SSDBCODI + MDS cycle. Called by `FeedbackService.apply_constraint` after channel updates, and directly by `ControlPanel` when the user clicks "Run clustering". Returns a serialisable response dict with all point positions, cluster IDs, and summary counts.

---

### 5.4 Domain Layer (Core Algorithms)

**Files:** `app/domain/`

This layer is the research-critical heart of the project. It has **zero I/O** — no Flask, no file reads, no network calls. Every module can be exercised by the 25 unit tests in `tests/` without a running server.

#### `clustering/`

| Module | Responsibility |
|---|---|
| `distance.py` | `MahalanobisDistance` — wraps an M matrix, exposes `pairwise(X)` returning an (n,n) distance matrix. Default is identity (Euclidean). |
| `ssdbscan.py` | Algorithm 1 from the paper. Prim-like min-reachability traversal from each labeled normal point. Returns `cluster_assignments`, raw distance matrix, reachability matrix, core distances. |
| `scores.py` | Three outlier-likelihood scores: `rScore` (reachability from normals), `lScore` (local density), `simScore` (similarity to outliers), combined into `tScore`. |
| `ssdbcodi.py` | Algorithm 2: orchestrates SSDBSCAN + scores + RandomForest classifier to assign every point a cluster label and outlier flag. |

#### `metric_learning/`

| Module | Responsibility |
|---|---|
| `base.py` | Abstract `MetricLearner` interface — `update(X, ...)`, `get_M()`, `reset()` |
| `itml_learner.py` | Information-Theoretic Metric Learning. Accepts similar/dissimilar pairs, projects M to the PSD cone, updates M by KL-minimisation with pairwise constraints. |
| `triplet_learner.py` | Single SGD step on a triplet `(anchor, positive, negative)`. Updates M so `d(anchor, positive) < d(anchor, negative)` with margin, projects M to PSD cone. |
| `composite.py` | Owns the canonical M. Routes each constraint to the right sub-learner, then syncs M back into both so they remain consistent. Handles `feature_hint` directly by editing the M diagonal. |

#### `constraints/`

| Module | Responsibility |
|---|---|
| `schemas.py` | Typed dataclasses: `MustLink`, `CannotLink`, `Triplet`, `ClusterCount`, `OutlierLabel`, `FeatureHint`, `ClusterMerge`, `Reassign`. Each has `to_dict()` / `constraint_from_dict()`. |
| `router.py` | `route_constraint(c) → ChannelType` — returns `LABEL`, `METRIC`, `BOTH`, or `NONE`. |
| `validators.py` | Validates a constraint object against the live session (point count, cluster existence, etc.). Returns `(ok: bool, message: str)`. |

#### `intent/`

| Module | Responsibility |
|---|---|
| `intent_types.py` | `IntentType` enum: `MUST_LINK`, `CANNOT_LINK`, `TRIPLET`, `CLUSTER_COUNT`, `OUTLIER_LABEL`, `FEATURE_HINT`, `CLUSTER_MERGE`, `REASSIGN`. |
| `rule_classifier.py` | Regex patterns ordered from most-specific to least. Returns `(IntentType, slots_dict)` or `(None, {})`. |
| `llm_classifier.py` | Builds a prompt from system_prompt.txt + few_shot_examples.json, calls the LLM, parses the JSON response robustly (strips markdown fences, finds balanced `{}` block). |

#### `projection/`

`mds_projector.py` — Wraps scikit-learn `MDS` with `dissimilarity="precomputed"`. Uses the current M to compute a Mahalanobis distance matrix, then projects to 2D. Called after every pipeline run.

---

### 5.5 Infrastructure Layer (External Boundaries)

**Files:** `app/infrastructure/`

Everything that touches the outside world lives here behind abstract base classes. Swapping any implementation is a `.env` change.

#### `llm/`

- `base.py` — `LLMClient` ABC: `chat(messages) → str`
- `ollama_client.py` — calls `http://localhost:11434/api/chat` (default model: `mistral-small3.1:latest`)
- `openai_client.py` — calls any OpenAI-compatible endpoint via `OPENAI_API_BASE` and `OPENAI_API_KEY`
- `factory.py` — reads `LLM_PROVIDER` from `.env`, returns the right client

#### `data/`

- `base.py` — `DataLoader` ABC: `load(path) → DataFrame`, `load_both(path) → (raw_df, normalised_df)`, `validate(df) → [warnings]`
- `csv_loader.py` — reads CSV/TSV, stores raw values, then z-score normalises numeric columns
- `factory.py` — selects loader by file extension

#### `storage/`

- `base.py` — `SessionStore` ABC: `get`, `save`, `delete`, `list_ids`, `rollback`
- `memory_store.py` — in-process Python dict (default; lost on restart)
- `pickle_store.py` — serialises `SessionState` to `data/sessions/<id>.pkl` (survives restart)

#### `debug/`

- `logger.py` — structured logging with `structlog`
- `debug_recorder.py` — when `DEBUG_DUMP_ENABLED=true`, writes per-iteration snapshots to `data/debug/<session_id>/iter_NNNN/` containing M, clusters, outliers, projection, scores, and the triggering constraint. Zero performance cost when disabled.
- `debug_tools.py` — notebook-friendly helpers to load and inspect debug dumps.

---

## 6. The SSDBCODI Algorithm Explained

SSDBCODI runs every time a constraint is applied. Here is what happens step by step:

### Step 1 — Cold start check

If `DN` (labeled normal points) is empty, SSDBCODI cannot expand. It returns every point as cluster 0, user-flagged outliers as -1, and placeholder scores. This lets the scatterplot render immediately on first load before the user has given any guidance.

### Step 2 — Precompute distances

`MahalanobisDistance(M).pairwise(X)` computes the (n×n) Mahalanobis distance matrix in one NumPy call:

```
d_M(p, q) = sqrt((p-q)^T · M · (p-q))
```

When M is the identity matrix this is Euclidean distance. As the user adds constraints, M deforms the space so that "important" features stretch distances and "unimportant" features shrink them.

### Step 3 — SSDBSCAN expansion

For each labeled normal point in `DN`, run a Prim-like minimum-spanning-tree traversal over the reachability distance matrix:

```
rDist(p, q) = max(cDist(p), cDist(q), dist(p, q))

where cDist(p) = distance to the MinPts-th nearest neighbour of p
```

Expansion stops when a point with a *different* label is encountered. At that point, the algorithm back-traces to find the longest edge along the path and assigns everything before that edge to the root's cluster. This lets clusters follow non-convex shapes and naturally separates when they would merge across a low-density bridge.

### Step 4 — Compute the three outlier scores

| Score | Meaning | High value means |
|---|---|---|
| `rScore(q)` | `exp(−min rDist to any labeled normal)` | Point is easy to reach from a normal cluster → probably normal |
| `lScore(q)` | `exp(−avg rDist to MinPts nearest neighbours)` | Point sits in a dense neighbourhood → probably normal |
| `simScore(q)` | `exp(−min dist to any labeled outlier)` | Point is close to a known outlier → probably outlier |
| `tScore(q)` | `α·(1−rScore) + β·(1−lScore) + γ·simScore` | Combined: high = probably outlier |

Default hyperparameters: α = 0.4, β = 0.4, γ = 0.2.

### Step 5 — Build Reliable Normal (RN) and Reliable Outlier (RO) sets

- **RN** = all points SSDBSCAN successfully assigned to a cluster, minus any user-flagged outliers
- **RO** = user-flagged outliers `DO` + the top-k unclustered points ranked by `tScore` (default k = 10)

### Step 6 — Train a RandomForest classifier

Training data: all points in RN (label = cluster_id, weight = rScore) + all points in RO (label = −1, weight = tScore). The score-weighted training means high-confidence assignments dominate.

### Step 7 — Predict and override

The classifier predicts a label for every point. Then explicit user labels override:
- Points in `DO` → label −1, `is_outlier = True`
- Points in `DN` → their explicit cluster_id
- Classifier predictions fill in the rest

---

## 7. The Chat / Intent Pipeline

```
User text + selected_ids
         │
         ▼
  RuleClassifier.classify(text)
  ┌──────────────────────────────────────────┐
  │  8 regex patterns, ordered:              │
  │  1. cluster_count  ("split into 3 ...")  │
  │  2. cluster_merge  ("merge cluster 1 …") │
  │  3. must_link      ("same class / group")│
  │  4. cannot_link    ("should not be …")   │
  │  5. triplet        ("more similar to …") │
  │  6. outlier_label  ("outlier / anomaly") │
  │  7. feature_hint   ("column is not …")   │
  └──────────────────────────────────────────┘
         │ (intent, slots) or (None, {})
         ▼
  Match found?  ─── No ──► LLMIntentClassifier
         │                        │
        Yes                       │
         │                        │
         ▼                        ▼
  Build constraint dict       System prompt:
  from slots + selected_ids   - intent type table
                              - 10 output formats
                              - selected_ids injected
                              + few-shot examples
                              + chat history
                                   │
                              LLM call (Ollama / OpenAI)
                                   │
                              Parse JSON from response
                                   │
         ┌─────────────────────────┘
         ▼
  validate(constraint, n_points)
         │
   valid? ─── No ──► return follow-up question to frontend
         │
        Yes
         │
         ▼
  return {complete: true, constraint: {...}}
  frontend auto-submits to /api/feedback/submit
```

The rule classifier covers the most common phrasings and avoids an LLM round-trip (typically 200–800 ms) for those cases. The LLM takes over for:
- Vague or ambiguous requests
- Complex relative comparisons (triplets)
- Feature name references that need semantic matching
- Off-topic inputs (the LLM politely redirects)

The LLM is prompted to return **only JSON** with a strict schema. The parser strips markdown fences and finds the first balanced `{...}` block, making it robust to models that add explanation text around the JSON.

---

## 8. The Constraint System

### The 8 Constraint Types

```
┌─────────────────┬──────────────────────────────────────────────┬─────────┬────────┐
│ Type            │ Meaning                                      │ Label   │ Metric │
├─────────────────┼──────────────────────────────────────────────┼─────────┼────────┤
│ must_link       │ Selected points belong to the same cluster   │    ✓    │   ✓    │
│ cannot_link     │ group_a and group_b must be in diff clusters │         │   ✓    │
│ triplet         │ anchor closer to positive than to negative   │         │   ✓    │
│ cluster_count   │ Target number of clusters (scope: all/sel/…) │    ✓    │        │
│ outlier_label   │ Mark/unmark points as outliers               │    ✓    │        │
│ feature_hint    │ Increase/decrease/ignore a feature's weight  │         │   ✓    │
│ cluster_merge   │ Merge two or more clusters into one          │    ✓    │        │
│ reassign        │ Move specific points to a target cluster     │    ✓    │        │
└─────────────────┴──────────────────────────────────────────────┴─────────┴────────┘
```

### Constraint Lifecycle

```
User intent (text + selection)
         │
  ChatService → constraint dict (JSON)
         │
  constraint_from_dict() → typed dataclass
         │
  validate() → check point IDs exist, cluster IDs exist, etc.
         │
  FeedbackService.apply_constraint()
    ├── state.snapshot()  ← undo checkpoint
    ├── route_constraint() → ChannelType
    ├── _apply_to_label_channel()  ← modifies DN / DO
    ├── _apply_to_metric_channel() ← updates M via learners
    └── pipeline_service.apply_constraint_and_recluster()
          ├── state.constraints_history.append(constraint)
          └── run_full_pipeline() → updated scatterplot
```

---

## 9. The Metric Learning System

The Mahalanobis matrix M is the geometric heart of the system. It starts as the identity (Euclidean distance) and deforms over time as the user adds constraints.

### CompositeMetricLearner

Owns the canonical M and keeps two sub-learners in sync:

```
CompositeMetricLearner
├── M  (n_features × n_features PSD matrix)
├── ITMLLearner
│   └── own copy of M, updated by must_link / cannot_link
└── TripletLearner
    └── own copy of M, updated by triplet constraints
```

After every update, `_sync_M()` copies the canonical M into both sub-learners so they agree before the next update.

### ITML (Information-Theoretic Metric Learning)

Used for pairwise constraints (`must_link`, `cannot_link`):
- Takes a set of pairs labelled +1 (similar) or −1 (dissimilar)
- Minimises KL-divergence from a Gaussian prior (current M) subject to pairwise distance constraints
- Result is always positive semi-definite by construction

### Triplet SGD

Used for `triplet` constraints `(anchor, positive, negative)`:
- One gradient step: push `d_M(anchor, positive)` below `d_M(anchor, negative) − margin`
- Projects M back to the PSD cone via eigenvalue clipping after each step

### Feature Hint Direct Edit

For `feature_hint` constraints, no optimisation is needed — the diagonal entry for the named feature is scaled directly:

| magnitude | scale factor | direction=decrease effect |
|---|---|---|
| slight | 0.70 | feature contributes 70% as much |
| moderate | 0.40 | feature contributes 40% as much |
| strong | 0.10 | feature contributes 10% as much |
| (any) | — | direction=ignore zeros the row+column |

After every direct edit, M is re-symmetrised and projected to the PSD cone.

---

## 10. Session State and Undo

Each session is a `SessionState` dataclass holding all mutable state for one analysis run:

```python
SessionState
├── session_id          # UUID
├── dataset             # z-scored DataFrame (used by all algorithms)
├── raw_dataset         # original DataFrame (used for CSV export)
├── M                   # current Mahalanobis matrix
├── DN                  # Dict[point_idx, cluster_id] — labeled normals
├── DO                  # Set[point_idx] — labeled outliers
├── constraints_history # List[Constraint]
├── current_clusters    # ndarray (n,) — latest cluster assignments
├── current_outliers    # ndarray (n,) bool — latest outlier flags
├── current_projection  # ndarray (n,2) — MDS coordinates
├── current_scores      # Dict[str, ndarray] — rscore/lscore/simscore/tscore
├── chat_history        # List[{role, content}]
└── _snapshots          # List[Dict] — up to 10 undo checkpoints
```

**Undo** works by snapshotting all mutable fields before every `apply_constraint` call, then rolling back to the last snapshot. The `FeedbackService` also drops its cached `CompositeMetricLearner` on undo so the next constraint starts from the rolled-back M rather than the forward-progressed one.

---

## 11. Data Flow: From User Message to Updated Scatterplot

The following is an annotated trace of the complete round-trip for the message *"these points are all the same class"* with 5 points selected:

```
1. Chatbox._handleSend()
   POST /api/chat/message
   { session_id: "abc123", text: "these points are all the same class",
     selected_ids: [3, 12, 47, 88, 102] }

2. routes_chat.send_message() → chat_service.process_message()

3. RuleClassifier checks text:
   Pattern "same class|one class|belong together" → MATCH
   intent = MUST_LINK, slots = {}

4. _build_constraint_from_rule():
   len(selected_ids) = 5 ≥ 2 → OK
   constraint_dict = { type: "must_link", point_ids: [3,12,47,88,102],
                        confidence: "explicit", source: "rule" }

5. validate(constraint, n_points=500) → OK

6. Response to frontend:
   { complete: true, intent: "must_link",
     constraint: { type: "must_link", point_ids: [...] },
     assistant_message: "Got it — marking these 5 points as the same cluster." }

7. Chatbox shows confirmation, auto-submits:
   POST /api/feedback/submit
   { session_id: "abc123", constraint: { type: "must_link", ... } }

8. routes_feedback.submit_constraint() → feedback_service.apply_constraint()

9. state.snapshot()  ← undo saved

10. route_constraint(MustLink) → ChannelType.BOTH

11. Label channel:
    existing DN labels on [3,12,47,88,102] → none found
    next_cluster_id = max(existing) + 1 = 3
    DN[3]=3, DN[12]=3, DN[47]=3, DN[88]=3, DN[102]=3
    None are in DO so nothing to discard

12. Metric channel:
    generate all-pairs: (3,12),(3,47),(3,88),(3,102),(12,47),(12,88),
                        (12,102),(47,88),(47,102),(88,102) — 10 pairs, all label=+1
    ITMLLearner.update(X, pairs, labels=[1,1,...,1])
    → new M returned, synced into TripletLearner

13. session_service.save(state)

14. pipeline_service.apply_constraint_and_recluster()
    state.M = new_M
    state.constraints_history.append(constraint)

15. run_full_pipeline():
    distance_func = MahalanobisDistance(state.M)
    SSDBCODI.fit(X, DN={...,3:3,12:3,...}, DO={...}, distance_func)
      → ssdbscan expands from all 5 newly labeled points
      → scores computed
      → RandomForest trained on RN+RO
      → all 500 points predicted
    MDSProjector.project(X, distance_func) → 500×2 coordinates

16. state.current_clusters updated
    state.current_projection updated
    session_service.save(state)

17. _build_response() → 500-element array of {id,x,y,cluster,is_outlier}

18. JSON response: { ready:true, points:[...], n_clusters:4, n_outliers:7,
                      n_constraints:1 }

19. AppState.setProjection(data) → emit("projection_changed")

20. Scatterplot.render(points)
    D3 updates 500 circles:
    - points 3,12,47,88,102 now coloured cluster 3 (teal)
    - rest redistributed by SSDBCODI result
    - legend refreshes
```

Total elapsed time (typical): ~300–800 ms depending on dataset size and LLM latency. For the rule-matched path the LLM is not involved at all, so it's usually under 400 ms.

---

## 12. File Map

```
metric-LLM-dashboard/
├── run.py                      Flask app entry point
├── config/
│   ├── config.py               Config dataclass (reads .env)
│   └── prompts/
│       ├── system_prompt.txt   LLM system prompt (constraint schema + rules)
│       └── few_shot_examples.json  Input/output examples for the LLM
├── app/
│   ├── __init__.py             Flask factory: registers blueprints + services
│   ├── api/
│   │   ├── routes_data.py      Upload / load sample / dataset info
│   │   ├── routes_cluster.py   Trigger pipeline run
│   │   ├── routes_chat.py      Process chat message
│   │   ├── routes_feedback.py  Submit constraint / undo
│   │   ├── routes_session.py   Session list / reset
│   │   ├── routes_debug.py     Debug inspection endpoints
│   │   ├── routes_export.py    CSV export
│   │   └── errors.py           ValidationError / NotFoundError → JSON
│   ├── models/
│   │   └── session_state.py    SessionState dataclass + snapshot/rollback
│   ├── services/
│   │   ├── chat_service.py     Rule → LLM → constraint classification
│   │   ├── feedback_service.py Apply constraint to label+metric channels
│   │   ├── pipeline_service.py SSDBCODI + MDS orchestration
│   │   └── session_service.py  CRUD wrapper around storage backend
│   ├── domain/
│   │   ├── clustering/
│   │   │   ├── distance.py     MahalanobisDistance + make_distance()
│   │   │   ├── ssdbscan.py     Algorithm 1: density expansion from seeds
│   │   │   ├── scores.py       rScore, lScore, simScore, tScore
│   │   │   └── ssdbcodi.py     Algorithm 2: full pipeline + classifier
│   │   ├── metric_learning/
│   │   │   ├── base.py         MetricLearner ABC
│   │   │   ├── itml_learner.py ITML for pairwise constraints
│   │   │   ├── triplet_learner.py SGD for triplet constraints
│   │   │   └── composite.py    Owns M, routes to sub-learners
│   │   ├── constraints/
│   │   │   ├── schemas.py      8 typed dataclasses + constraint_from_dict
│   │   │   ├── router.py       route_constraint → ChannelType
│   │   │   └── validators.py   validate(constraint, n_points) → (bool, str)
│   │   ├── intent/
│   │   │   ├── intent_types.py IntentType enum
│   │   │   ├── rule_classifier.py Regex fast-path
│   │   │   └── llm_classifier.py  LLM fallback + JSON parser
│   │   └── projection/
│   │       └── mds_projector.py   Precomputed-dissimilarity MDS → 2D
│   └── infrastructure/
│       ├── llm/
│       │   ├── base.py         LLMClient ABC
│       │   ├── ollama_client.py Local Ollama
│       │   ├── openai_client.py OpenAI-compatible remote
│       │   └── factory.py      Choose by LLM_PROVIDER in .env
│       ├── data/
│       │   ├── base.py         DataLoader ABC
│       │   ├── csv_loader.py   Load + z-score normalise CSV/TSV
│       │   └── factory.py      Choose by file extension
│       ├── storage/
│       │   ├── base.py         SessionStore ABC
│       │   ├── memory_store.py In-process dict (default)
│       │   └── pickle_store.py Persistent file-backed store
│       └── debug/
│           ├── logger.py       structlog setup
│           ├── debug_recorder.py Dump iteration snapshots to disk
│           └── debug_tools.py  Notebook helpers for debug dumps
├── static/
│   ├── js/
│   │   ├── main.js             App bootstrap + wiring
│   │   ├── state.js            AppState pub/sub
│   │   ├── api_client.js       fetch() wrappers for every endpoint
│   │   ├── components/
│   │   │   ├── control_panel.js Upload/run/undo/export controls
│   │   │   ├── scatterplot.js  D3 SVG + lasso + click selection
│   │   │   ├── chatbox.js      Chat UI + auto-submit
│   │   │   └── legend.js       Cluster colour legend
│   │   └── utils/
│   │       └── colors.js       Stable categorical colour palette
│   ├── css/
│   │   ├── main.css            Layout
│   │   ├── scatterplot.css     Dot styles, outlier dashes, selection ring
│   │   └── chatbox.css         Message bubbles, context bar
│   └── lib/
│       ├── d3.v7.min.js
│       └── d3-lasso.min.js
├── templates/
│   └── index.html              Single-page app shell
├── data/
│   ├── samples/                blobs.csv, circles.csv, moons.csv, wine.csv
│   ├── sessions/               pickle files (if pickle store enabled)
│   ├── uploads/                temporary uploaded CSVs
│   └── debug/                  per-session / per-iteration debug dumps
├── tests/
│   ├── conftest.py             Shared fixtures (small X, DN, DO)
│   ├── test_ssdbcodi.py
│   ├── test_distance.py
│   ├── test_metric_learner.py
│   ├── test_rule_classifier.py
│   └── test_constraints.py
├── requirements.txt
├── scripts/download_libs.sh    Downloads D3 and d3-lasso into static/lib/
├── SETUP.md
└── README.md
```

---

*Document generated from code review of the April 2026 commit.*
