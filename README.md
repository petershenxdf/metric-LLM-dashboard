# SSDBCODI Interactive Dashboard

An interactive dashboard that combines:
- **SSDBCODI** (semi-supervised density-based clustering with outlier detection)
- **Metric learning** (ITML + triplet SGD) driven by user feedback
- **LLM-powered chatbox** (Ollama, default model `mistral-small3.1:latest`) that turns natural language into structured constraints

## Quick start

```bash
# 1. clone / unzip the project, cd into it
cd ssdbcodi-dashboard

# 2. create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. install dependencies
pip install -r requirements.txt

# 4. install + start Ollama, then pull the model
# (see https://ollama.com for installer)
ollama pull mistral-small3.1:latest
ollama serve                       # leave this running in another terminal

# 5. copy the example env file
cp .env.example .env

# 6. run the dashboard
python run.py

# 7. open http://localhost:5000 in your browser
```

## Usage flow

1. Click **Load sample** or **Upload CSV** in the control panel.
2. Click **Run clustering** — the scatterplot fills with colored points (clusters) and gray-edged dots (outliers).
3. Use the **lasso** (drag on the scatterplot) to select points.
4. Type instructions in the **chatbox**:
   - "These selected points should be one class"
   - "These are outliers"
   - "The unselected points should split into 3 groups"
   - "Color feature is not important"
5. The chatbox confirms its understanding, then re-runs the pipeline with the new constraint.
6. The scatterplot updates. Repeat.

## Project structure

See `docs/architecture.md` (or the diagram you were given). Key folders:

```
config/         # all tunable parameters
app/api/        # Flask routes (thin)
app/services/   # orchestration
app/domain/    # SSDBCODI, metric learner, constraints — pure algorithms, no I/O
app/infrastructure/  # LLM client, data loader, session store — replaceable
static/         # frontend (D3.js + vanilla JS)
templates/      # index.html
tests/          # unit + integration tests
```

## Switching the LLM

Edit `.env`:

```
LLM_PROVIDER=ollama
LLM_MODEL=mistral-small3.1:latest
LLM_BASE_URL=http://localhost:11434
```

To switch to OpenAI-compatible:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
```

No code changes required.

## Running tests

```bash
pytest tests/ -v
```
