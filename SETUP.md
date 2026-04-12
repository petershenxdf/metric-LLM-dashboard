# Setup walkthrough

Step-by-step instructions for running the SSDBCODI Interactive Dashboard locally.
Estimated time: 10-15 minutes the first time.

## Prerequisites

- [Anaconda](https://www.anaconda.com/download) (Python 3.10 or newer — included with Anaconda)
- An internet connection (once, for the initial library downloads and the Ollama model pull)
- ~4 GB of free disk space (most of it for the LLM model)

All commands below should be run in **Anaconda Prompt** (search "Anaconda Prompt" in the Start menu).

## Step 1 — Unpack the project

Right-click `ssdbcodi-dashboard.zip` in Windows Explorer and select **Extract All**, then open Anaconda Prompt and navigate to the extracted folder:

```
cd path\to\ssdbcodi-dashboard
```

## Step 2 — Create a Conda environment

```
conda create -n ssdbcodi python=3.10 -y
conda activate ssdbcodi
```

You should see `(ssdbcodi)` prepended to your prompt.

## Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs Flask, numpy, scikit-learn, metric-learn, pandas, and a few others.
Takes 1-3 minutes depending on your connection.

## Step 4 — Download the frontend JavaScript libraries

The project ships with placeholder files for D3 and d3-lasso. Anaconda Prompt has `curl` built in, so run:

```
curl -o static\lib\d3.v7.min.js https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js
curl -o static\lib\d3-lasso.min.js https://cdn.jsdelivr.net/npm/d3-lasso@0.0.5/build/d3-lasso.min.js
```

This downloads `d3.v7.min.js` and `d3-lasso.min.js` into `static\lib\`.

Verify the files are real (not placeholders) by checking their sizes in Anaconda Prompt:

```
dir static\lib\*.js
```

`d3.v7.min.js` should be ~300 KB or more; `d3-lasso.min.js` ~10 KB or more. If they are only a few KB, the download failed — re-run the curl commands above.

## Step 5 — Install and start Ollama

Download Ollama for your OS from https://ollama.com and install it.

Pull the default model used by the dashboard:

```bash
ollama pull mistral-small3.1:latest
```

This is about 15 GB and will take a while the first time. Grab a coffee.

Start the Ollama server (keep this running in a separate terminal):

```bash
ollama serve
```

You should see output like `Listening on 127.0.0.1:11434`.

Verify it's working in Anaconda Prompt:

```
curl http://localhost:11434/api/tags
```

You should see a JSON list containing `mistral-small3.1:latest`.

## Step 6 — Configure environment variables

```
copy .env.example .env
```

The default values in `.env` already point to local Ollama on port 11434 with the
`mistral-small3.1:latest` model, so you usually don't need to edit anything.

If you want to use a different LLM (e.g. GPT-4o-mini), edit `.env`:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key-here
```

No code changes needed — the LLM factory reads these at startup.

## Step 7 — Run the tests (optional but recommended)

```bash
pytest tests/ -v
```

You should see `25 passed`. If any tests fail, stop and debug before continuing.

## Step 8 — Start the dashboard

```bash
python run.py
```

You should see Flask output like:

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

## Step 9 — Open the UI

Open your browser at **http://localhost:5000**.

You'll see:

- A header with a status indicator (gray = idle, green = ready, orange = working)
- A left panel with a control bar, a big empty scatterplot area, and a legend row
- A right panel with the chatbox

## Step 10 — First run: load a sample and cluster

1. Click the **Load sample** dropdown in the control bar, pick **blobs** (the simplest dataset).
2. The info text updates to show `212 points, 2 features`.
3. Click the blue **Run clustering** button.
4. After a second or two, colored dots fill the scatterplot. Because you have not
   given any labels yet, everything starts as a single cluster (all one color).
5. The legend row shows `Cluster 0 (212)`.

## Step 11 — Select points and talk to the chatbox

1. **Drag** on the scatterplot to draw a lasso around a group of points in one
   corner of the blob. They get a dark outline to show they are selected. The
   chatbox header updates to show `Selected: N points`.
2. In the chatbox, type: `these points are one class`
3. Press Enter.
4. The chatbox echoes a confirmation (`Got it -- marking these N points as the
   same cluster.`) and the scatterplot re-renders. The selected points now have
   a new color, and other clusters may shift.
5. Select another group of points and type: `these should be a different class`
6. Continue building up clusters this way.

## Step 12 — Label outliers

1. Lasso a few stray points far from any cluster.
2. Type `these are outliers`.
3. The selected points get gray fills with dashed red borders. The legend adds
   an `Outliers` row.

## Step 13 — Try a feature hint

1. In the chatbox, type: `ignore the color feature` (for the blobs dataset
   which has only `x1` and `x2`, try `x2 is not important` instead).
2. The LLM will ask a follow-up about how aggressively to reduce it.
3. Answer (e.g. `a lot`) — the metric learner will update M and the scatterplot
   re-projects with the new distance function.

## Step 14 — Undo and reset

- The **Undo** button rolls back the last constraint (up to 10 steps).
- The **Reset** button wipes all labels, M, and constraints, and re-runs from scratch.

## Trying more datasets

- **moons** (320 points, 2D) — two interleaved half-moons plus noise. SSDBCODI
  shines here because it handles non-convex shapes. Try labeling one point in
  each moon with `these two points are different classes`.
- **circles** (315 points, 2D) — concentric rings. Even more non-convex.
- **wine** (186 points, 13 features) — real chemistry data. Good for
  testing feature_hint: try `alcohol is not important`.

## Using your own data

1. Prepare a CSV file. The first row should be column headers. All columns
   should be numeric — the loader will drop non-numeric columns and z-score
   normalize the numeric ones.
2. Click **Upload CSV** in the control bar and select your file.
3. Proceed from Step 10.

Constraints:
- Max 20 MB (configurable via `MAX_UPLOAD_SIZE_MB` in `.env`).
- Datasets above ~5000 points will be slow — sub-sample for interactive use.
- Datasets above ~100 features should be PCA-reduced beforehand.

## Troubleshooting

**The page loads but the scatterplot stays empty**
Check the browser console. The most common cause is the placeholder
`d3.v7.min.js` or `d3-lasso.min.js`. Re-run the curl commands from Step 4.

**Chat returns "I had trouble understanding that"**
The LLM is unreachable. Check:
1. `curl http://localhost:11434/api/tags` returns a response
2. `.env` has `LLM_BASE_URL=http://localhost:11434`
3. The model is pulled: `ollama list` should show `mistral-small3.1:latest`

**Clustering is very slow**
Either your dataset is too big (>5000 points) or too high-dimensional
(>100 features). MDS is O(n^3) in the worst case. Reduce the data first.

**ITML warnings in the terminal**
ITML occasionally fails to converge on small or degenerate pair sets. The code
catches this and keeps the previous M, so the dashboard stays responsive.
Nothing to fix.

**`ModuleNotFoundError: No module named 'app'`**
You are running `python run.py` from the wrong directory. Always run from
the project root.

**Port 5000 is already in use**
Edit `.env` and change `PORT=5000` to something else.

## Exporting your labeled dataset

When you have finished assigning clusters and marking outliers, click the
**Export CSV** button in the control bar. Your browser will download a file
named `<original_name>_labeled.csv` containing:

- All original numeric columns (pre-normalization, in their real units)
- `point_id` column (the row index used throughout the dashboard)
- `cluster_label` (integer; -1 means outlier)
- `is_outlier` (True / False)
- `rscore`, `lscore`, `simscore`, `tscore` (the SSDBCODI outlierness scores)

The export endpoint is also available directly:

```
GET /api/export/csv/<session_id>?include_scores=true
```

Set `include_scores=false` if you only want the labels and not the scores.

## Debugging: saving intermediate results

The dashboard can dump a full snapshot of every pipeline iteration to disk
so you can inspect what changed between constraint applications.

Enable it in `.env`:

```
DEBUG_DUMP_ENABLED=true
DEBUG_DUMP_DIR=./data/debug
LOG_LEVEL=DEBUG
```

Restart the app. Each iteration now writes to:

```
data/debug/<session_id>/iter_NNNN/
    meta.json        -- iteration number, timestamp, duration, triggering constraint
    M.npy            -- Mahalanobis matrix at this iteration
    clusters.npy     -- cluster label per point
    outliers.npy     -- outlier flag per point
    projection.npy   -- 2D MDS coordinates
    scores.npz       -- rscore, lscore, simscore, tscore
    labels.json      -- DN and DO sets
```

Inspection options:

**1. From the browser / curl:**
```bash
curl http://localhost:5000/api/debug/iterations/<session_id>
curl http://localhost:5000/api/debug/iteration/<session_id>/3
```

**2. From a notebook or script:**
```python
from app.infrastructure.debug.debug_tools import load_iteration, diff_iterations

it = load_iteration("data/debug/abc123/iter_0007")
print(it["meta"])                # iteration metadata
print(it["M"].shape)              # (d, d) numpy array
print(it["scores"]["tscore"])     # score vector
print(it["clusters"])             # cluster label vector

# What changed between two iterations?
diff = diff_iterations(
    "data/debug/abc123/iter_0003",
    "data/debug/abc123/iter_0004",
)
print(diff)
# {'M_frobenius_diff': 0.42, 'n_relabeled': 17, 'n_outlier_flips': 2, ...}
```

**3. Plot the projection from a dump:**
```python
import matplotlib.pyplot as plt
from app.infrastructure.debug.debug_tools import load_iteration

it = load_iteration("data/debug/abc123/iter_0005")
plt.scatter(it["projection"][:, 0], it["projection"][:, 1], c=it["clusters"])
plt.title(f"Iteration 5 (after {it['meta']['triggering_constraint']['type']})")
plt.show()
```

Clear a session's dumps:
```
POST /api/debug/clear/<session_id>
```

Debug dumps are independent of session storage -- they survive app restarts
and can be inspected offline at any time.

## Where to look when you want to change things

| I want to... | Edit this file |
|---|---|
| Change the default model | `.env` → `LLM_MODEL` |
| Swap LLM provider | `.env` → `LLM_PROVIDER` (ollama / openai) |
| Change SSDBCODI hyperparameters | `.env` → `SSDBCODI_*` variables |
| Enable / disable debug dumps | `.env` → `DEBUG_DUMP_ENABLED` |
| Change debug dump location | `.env` → `DEBUG_DUMP_DIR` |
| Adjust log verbosity | `.env` → `LOG_LEVEL` (DEBUG / INFO / WARNING) |
| Edit the LLM prompt | `config/prompts/system_prompt.txt` |
| Add few-shot examples | `config/prompts/few_shot_examples.json` |
| Add a new constraint type | `app/domain/constraints/schemas.py` + `router.py` |
| Add a new data format (e.g. Parquet) | Create new loader in `app/infrastructure/data/`, register in `factory.py` |
| Add a new LLM provider | Create new client in `app/infrastructure/llm/`, register in `factory.py` |
| Change cluster colors | `static/js/utils/colors.js` |
| Change the scatterplot layout | `static/js/components/scatterplot.js` |
| Add a new chatbox command pattern | `app/domain/intent/rule_classifier.py` |
| Customize the export CSV columns | `app/api/routes_export.py` |

## Running the tests

```bash
pytest tests/              # all tests
pytest tests/test_ssdbcodi.py -v   # just the algorithm
```

All tests are offline — they do not require Ollama to be running.

## When you are ready to deploy

The current setup (in-memory session store, Flask dev server) is fine for
local development but NOT for production. Before deploying:

1. Switch to `STORAGE_BACKEND=pickle` (or implement a Redis-backed store) so
   sessions survive worker restarts.
2. Run behind a real WSGI server: `gunicorn -w 1 'app:create_app()'` — note
   the `-w 1`, because the in-memory / pickle store is single-process.
3. Put it behind nginx or another reverse proxy for TLS and static file
   serving.
4. Set `FLASK_DEBUG=false` and `SECRET_KEY` to a real random value.
5. Limit `MAX_UPLOAD_SIZE_MB` to match your actual use case.
