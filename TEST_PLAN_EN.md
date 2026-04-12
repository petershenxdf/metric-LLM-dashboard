# SSDBCODI Interactive Dashboard — Comprehensive Manual Test Plan

> **Purpose**: Systematically test every user-facing feature of the dashboard as an end-user, covering data loading, visualization, all 8 constraint types, chat/LLM interaction, undo/reset, export, and edge cases.

---

## Prerequisites

| Item | Requirement |
|------|------------|
| Server | `python app.py` running at `http://127.0.0.1:5000` |
| LLM | Ollama running locally with `mistral-small3.1:latest` (or configured OpenAI key) |
| Browser | Chrome or Firefox, DevTools open (Console tab) to catch JS errors |
| Test CSV | Prepare a custom CSV with ~50 rows, 3 numeric + 1 text column (for upload testing) |
| Large CSV | Prepare a CSV > 20 MB (for size limit testing) |

---

## Module 1: Application Startup & Health

### T1.1 — Server Startup
- **Action**: Start the server, open `http://127.0.0.1:5000` in browser.
- **Goal**: Verify the page loads without errors.
- **Expected**: Two-panel layout appears. Left panel shows placeholder text "Upload a CSV or load a sample to begin". Right panel shows chatbox with "Assistant" header. Status dot is gray ("Idle").

### T1.2 — Health Check
- **Action**: Open `http://127.0.0.1:5000/health` in a new tab.
- **Goal**: Verify backend and LLM availability.
- **Expected**: JSON response with `status: "ok"` and LLM availability status.

### T1.3 — Initial Button States
- **Action**: Inspect all buttons on the control bar before loading data.
- **Goal**: Verify buttons are correctly disabled when no data is loaded.
- **Expected**: "Run clustering", "Undo", "Reset", "Export CSV" are all disabled. "Upload CSV" and "Load sample" are enabled.

---

## Module 2: Data Loading

### T2.1 — Load Sample: Blobs
- **Action**: Click "Load sample" dropdown → select "blobs".
- **Goal**: Verify sample dataset loads and initial clustering runs.
- **Expected**:
  - Info bar shows "212 points, 2 features".
  - Scatterplot renders 212 points with initial clustering.
  - Legend shows cluster labels with counts.
  - Status dot turns green ("Ready").
  - "Run clustering", "Reset", "Export CSV" buttons become enabled.

### T2.2 — Load Sample: Moons
- **Action**: Click "Load sample" → "moons".
- **Goal**: Test non-convex dataset loading.
- **Expected**: 320 points displayed. Two interleaved half-moon shapes visible. Clusters may initially be imperfect (this is expected — constraints will refine them).

### T2.3 — Load Sample: Circles
- **Action**: Click "Load sample" → "circles".
- **Goal**: Test concentric ring dataset.
- **Expected**: 315 points displayed. Concentric ring pattern visible in scatterplot.

### T2.4 — Load Sample: Wine
- **Action**: Click "Load sample" → "wine".
- **Goal**: Test high-dimensional dataset (13 features projected to 2D via MDS).
- **Expected**: 186 points displayed. Info bar shows "186 points, 13 features". Points are projected to 2D — visual clusters may not be as clearly separated as 2D datasets.

### T2.5 — Upload Custom CSV (Valid)
- **Action**: Click "Upload CSV" → select your test CSV with 3 numeric + 1 text column.
- **Goal**: Verify CSV upload, non-numeric column handling, and warnings.
- **Expected**:
  - Upload succeeds. Session created.
  - A warning message indicates the text column was dropped.
  - Only numeric columns are used for clustering.
  - Scatterplot renders correctly.

### T2.6 — Upload Oversized CSV
- **Action**: Click "Upload CSV" → select the >20 MB CSV file.
- **Goal**: Verify file size limit is enforced.
- **Expected**: Upload rejected. Error message displayed indicating file exceeds maximum size.

### T2.7 — Switch Datasets Mid-Session
- **Action**: Load "blobs", add a constraint, then load "moons".
- **Goal**: Verify session resets cleanly when switching datasets.
- **Expected**: Previous constraints and state are cleared. New dataset loads with fresh clustering. Chat history may or may not reset (verify behavior).

---

## Module 3: Visualization & Interaction

### T3.1 — Point Hover Tooltip
- **Action**: Load "blobs". Hover over any point.
- **Goal**: Verify tooltip displays correctly.
- **Expected**: Tooltip shows "id N | cluster X". If the point is an outlier, it also shows "(outlier)".

### T3.2 — Single Point Selection (Click)
- **Action**: Click on a single point in the scatterplot.
- **Goal**: Verify single selection works.
- **Expected**: Clicked point gets a dark outline. Chatbox selection context updates to "Selected: 1 point [idN]".

### T3.3 — Toggle Selection (Shift-Click)
- **Action**: Click point A. Then Shift-click point B. Then Shift-click point A again.
- **Goal**: Verify multi-select toggle behavior.
- **Expected**: After step 1: A selected. After step 2: A and B selected. After step 3: only B selected (A toggled off).

### T3.4 — Lasso Selection
- **Action**: Click and drag to draw a lasso polygon around a group of ~10 points.
- **Goal**: Verify lasso selection works.
- **Expected**: All enclosed points get dark outlines. Selection context shows "Selected: ~10 points [id1, id2, ...]".

### T3.5 — Large Lasso Selection (>8 points)
- **Action**: Lasso-select about 20 points.
- **Goal**: Verify display truncation for large selections.
- **Expected**: Selection context shows "Selected: 20 points" (without listing individual IDs, since count > 8).

### T3.6 — Clear Selection
- **Action**: Click on empty space in the scatterplot (not on any point).
- **Goal**: Verify selection clears.
- **Expected**: All point outlines removed. Selection context reverts to "No points selected."

### T3.7 — Legend Accuracy
- **Action**: Load "blobs", run clustering. Count points of each cluster color visually.
- **Goal**: Verify legend counts match actual point counts.
- **Expected**: Each "Cluster N (count)" in the legend matches the number of points with that color.

### T3.8 — Window Resize
- **Action**: Load a dataset, then resize the browser window.
- **Goal**: Verify responsive layout.
- **Expected**: Scatterplot re-renders to fit new dimensions. Points maintain relative positions.

---

## Module 4: Selection Groups

### T4.1 — Create a Selection Group
- **Action**: Load "blobs". Lasso 5 points. Click "Add selection as group".
- **Goal**: Verify group staging works.
- **Expected**: A chip labeled "Group A (5 points)" appears in the groups section. Selection clears from scatterplot. Selection context reverts to "No points selected."

### T4.2 — Create Multiple Groups
- **Action**: After T4.1, lasso 5 different points. Click "Add selection as group" again.
- **Goal**: Verify multiple groups can be staged.
- **Expected**: Two chips: "Group A (5 points)" and "Group B (5 points)".

### T4.3 — Remove a Group
- **Action**: Click the "x" button on Group A.
- **Goal**: Verify group removal and re-labeling.
- **Expected**: Group A removed. Remaining group re-labels to "Group A" (previously Group B).

### T4.4 — Clear All Groups
- **Action**: Create 2 groups, then click "Clear groups".
- **Goal**: Verify all groups are removed.
- **Expected**: All group chips disappear. Groups section shows hint text.

---

## Module 5: Constraint — Must-Link

### T5.1 — Must-Link via Direct Selection
- **Action**: Load "moons". Lasso 5 points from one visual moon. Type: `these are one class`. Send.
- **Goal**: Verify must-link constraint creation and queuing.
- **Expected**:
  - Chat shows user message and assistant confirmation: "Got it — these 5 points should be in the same cluster."
  - "Run clustering" button shows "(1 pending)" badge and pulses orange.
  - Constraint is queued (not yet applied).

### T5.2 — Run Must-Link
- **Action**: Click "Run clustering".
- **Goal**: Verify constraint is applied and clustering updates.
- **Expected**:
  - Status dot turns orange (working), then green (ready).
  - The 5 selected points are now in the same cluster (same color).
  - Metric M updated — nearby points may also change cluster.
  - Pending count returns to 0.

### T5.3 — Must-Link with Keyword Variations
- **Action**: Select points. Try each phrase:
  - `group these together`
  - `same cluster`
  - `these belong together`
  - `should be one group`
- **Goal**: Verify rule-based classifier recognizes all variations.
- **Expected**: Each phrase recognized as must_link intent without hitting the LLM.

### T5.4 — Must-Link Without Selection
- **Action**: Ensure no points selected. Type: `these are one class`. Send.
- **Goal**: Verify error handling when no points are selected.
- **Expected**: Follow-up question: "Please select some points first."

---

## Module 6: Constraint — Cannot-Link

### T6.1 — Cannot-Link via Two Groups
- **Action**: Load "blobs". Lasso 5 points from cluster region A → "Add selection as group". Lasso 5 points from cluster region B → "Add selection as group". Type: `these should not be together`. Send.
- **Goal**: Verify cannot-link with two staged groups.
- **Expected**:
  - Constraint created with group_a = [group A ids] and group_b = [group B ids].
  - Confirmation message displayed.
  - Constraint queued as pending.

### T6.2 — Run Cannot-Link
- **Action**: Click "Run clustering".
- **Goal**: Verify metric update pushes the groups apart.
- **Expected**: Points from Group A and Group B are assigned to different clusters. Distance metric between groups increased.

### T6.3 — Cannot-Link Keyword Variations
- **Action**: Stage two groups. Try:
  - `separate these`
  - `different classes`
  - `not the same group`
- **Goal**: Verify rule-based recognition.
- **Expected**: Each phrase triggers cannot_link intent.

### T6.4 — Cannot-Link with Only One Group
- **Action**: Stage only one group. Type: `these should be separate`. Send.
- **Goal**: Verify handling of insufficient groups.
- **Expected**: Follow-up question asking to stage a second group.

---

## Module 7: Constraint — Triplet

### T7.1 — Triplet via Three Groups (Single Points)
- **Action**: Load "blobs".
  1. Click one point near cluster center → "Add selection as group" (anchor, Group A).
  2. Click one point close to anchor → "Add selection as group" (positive, Group B).
  3. Click one point far from anchor → "Add selection as group" (negative, Group C).
  4. Type: `the anchor is more similar to the positive than the negative`. Send.
- **Goal**: Verify triplet constraint creation.
- **Expected**:
  - Constraint: `{type: "triplet", anchor: <A id>, positive: <B id>, negative: <C id>}`.
  - Confirmation message displayed.

### T7.2 — Run Triplet
- **Action**: Click "Run clustering".
- **Goal**: Verify metric learns the relative similarity.
- **Expected**: After re-projection, anchor and positive move closer together, anchor and negative move farther apart in the 2D visualization.

### T7.3 — Triplet Keyword Variations
- **Action**: Stage 3 single-point groups. Try:
  - `point A is closer to B than C`
  - `more like the second than the third`
- **Expected**: Recognized as triplet intent.

---

## Module 8: Constraint — Outlier Label

### T8.1 — Mark Points as Outliers
- **Action**: Load "circles". Lasso 3-4 points that appear isolated or between rings. Type: `these are outliers`. Send.
- **Goal**: Verify outlier labeling.
- **Expected**:
  - Confirmation: "Got it — marking N points as outliers."
  - Constraint queued.

### T8.2 — Run Outlier Label
- **Action**: Click "Run clustering".
- **Goal**: Verify outlier visualization.
- **Expected**:
  - Marked points turn gray with dashed red border.
  - Legend adds "Outliers (N)" row.
  - These points excluded from cluster assignments.

### T8.3 — Unmark Outliers
- **Action**: Select the previously marked outlier points. Type: `these are not outliers`. Send. Run clustering.
- **Goal**: Verify outlier un-labeling.
- **Expected**: Points return to normal cluster assignment and color. Outlier count decreases.

### T8.4 — Outlier Keyword Variations
- **Action**: Select points. Try:
  - `these are noise`
  - `anomalies`
  - `abnormal points`
- **Expected**: Each recognized as outlier_label intent with `is_outlier: true`.

---

## Module 9: Constraint — Cluster Count

### T9.1 — Global Cluster Count
- **Action**: Load "blobs". Type (no selection): `split into 3 groups`. Send.
- **Goal**: Verify global cluster_count constraint.
- **Expected**:
  - Constraint: `{type: "cluster_count", scope: "all", target_k: 3}`.
  - Confirmation displayed.

### T9.2 — Run Cluster Count
- **Action**: Click "Run clustering".
- **Goal**: Verify clustering respects target_k.
- **Expected**: Scatterplot shows exactly 3 clusters (3 distinct colors). Legend lists 3 clusters.

### T9.3 — Scoped Cluster Count (Selected)
- **Action**: Lasso half the points. Type: `the selected points should be 2 clusters`. Send. Run.
- **Goal**: Verify cluster_count scoped to selected points.
- **Expected**: Constraint has `scope: "selected"`. Selected region shows 2 distinct clusters.

### T9.4 — Scoped Cluster Count (Unselected)
- **Action**: Select a few points. Type: `the unselected points should be 3 groups`. Send. Run.
- **Goal**: Verify cluster_count scoped to unselected points.
- **Expected**: Constraint has `scope: "unselected"`.

### T9.5 — Cluster Count Keywords
- **Action**: Try:
  - `I want 4 classes`
  - `there should be 2 clusters`
  - `divide into 5 groups`
- **Expected**: Each recognized with correct target_k value.

---

## Module 10: Constraint — Feature Hint

### T10.1 — Feature Hint (Ignore)
- **Action**: Load "wine" (13 features). Type: `color_intensity is not important`. Send.
- **Goal**: Verify feature_hint intent recognition.
- **Expected**: LLM (or rule) asks follow-up: "Do you want to completely ignore the color_intensity feature, or just reduce its importance?"

### T10.2 — Complete Feature Hint
- **Action**: Reply: `completely ignore it`. Send.
- **Goal**: Verify two-turn constraint completion.
- **Expected**:
  - Constraint: `{type: "feature_hint", feature_name: "color_intensity", direction: "ignore", magnitude: "strong"}`.
  - Confirmation displayed. Constraint queued.

### T10.3 — Run Feature Hint
- **Action**: Click "Run clustering".
- **Goal**: Verify M matrix update and projection change.
- **Expected**: Scatterplot re-projects. Points that were only separated by color_intensity now appear closer together. Clusters may merge or shift.

### T10.4 — Feature Hint (Decrease)
- **Action**: Type: `reduce the importance of alcohol`. Then reply: `just slightly`. Run.
- **Goal**: Verify decrease direction with slight magnitude.
- **Expected**: Constraint: `{direction: "decrease", magnitude: "slight"}`. Projection changes subtly.

---

## Module 11: Constraint — Cluster Merge

### T11.1 — Merge Two Clusters
- **Action**: Load "blobs". Run initial clustering (ensure >=3 clusters). Note cluster IDs from legend (e.g., Cluster 0, Cluster 1, Cluster 2). Type: `merge cluster 0 and cluster 1`. Send.
- **Goal**: Verify cluster_merge constraint.
- **Expected**: Confirmation: "Got it — merging clusters 0 and 1 into one." Constraint queued.

### T11.2 — Run Cluster Merge
- **Action**: Click "Run clustering".
- **Goal**: Verify clusters are combined.
- **Expected**: Previous cluster 0 and cluster 1 points now share the same color. Legend shows one fewer cluster. Total point count unchanged.

### T11.3 — Merge Non-Existent Cluster
- **Action**: Type: `merge cluster 0 and cluster 99`. Send.
- **Goal**: Verify error handling for invalid cluster ID.
- **Expected**: Error or follow-up message indicating cluster 99 does not exist.

---

## Module 12: Constraint — Reassign

### T12.1 — Reassign Point to Different Cluster
- **Action**: Load "blobs". Run clustering. Click on a single point in cluster 2 (note its id from tooltip). Type: `move this point to cluster 0`. Send.
- **Goal**: Verify reassign constraint.
- **Expected**: Constraint: `{type: "reassign", point_ids: [<id>], target_cluster_id: 0}`. Confirmation displayed.

### T12.2 — Run Reassign
- **Action**: Click "Run clustering".
- **Goal**: Verify point moves to target cluster.
- **Expected**: The selected point now has the color of cluster 0. Cluster counts in legend update accordingly.

### T12.3 — Reassign Multiple Points
- **Action**: Lasso 5 points from cluster 1. Type: `move these to cluster 2`. Send. Run.
- **Goal**: Verify batch reassignment.
- **Expected**: All 5 points change to cluster 2's color.

---

## Module 13: Chat & LLM Interaction

### T13.1 — Rule-Based Fast Path
- **Action**: Load "blobs". Select 3 points. Type: `these are one class`. Send.
- **Goal**: Verify rule-based classifier handles common phrases quickly.
- **Expected**: Response is near-instant (no LLM latency). Constraint created correctly.

### T13.2 — LLM Fallback Path
- **Action**: Select 3 points. Type: `I think the selected points have something in common and should be grouped`. Send.
- **Goal**: Verify LLM handles ambiguous/complex phrasing.
- **Expected**: Response takes a few seconds (LLM processing). "Thinking..." indicator visible. Correct intent (must_link) extracted.

### T13.3 — Multi-Turn Conversation
- **Action**: Type: `the feature alcohol is not that relevant`. Send. Wait for follow-up. Reply: `reduce it moderately`. Send.
- **Goal**: Verify multi-turn intent completion.
- **Expected**: First turn returns follow-up question. Second turn completes the constraint.

### T13.4 — Off-Topic Message
- **Action**: Type: `what's the weather today?`. Send.
- **Goal**: Verify off-topic handling.
- **Expected**: Response: "I can only help you adjust the clustering. What would you like to do with the points?"

### T13.5 — Empty Message
- **Action**: Click "Send" with empty text area (or just whitespace).
- **Goal**: Verify empty input handling.
- **Expected**: Message not sent, or error message displayed. No crash.

### T13.6 — Very Long Message
- **Action**: Type a 500+ character message describing a complex clustering desire.
- **Goal**: Verify handling of long input.
- **Expected**: LLM processes it and extracts intent. No truncation error.

### T13.7 — Chat History Persistence
- **Action**: Send 5 messages, scroll up to verify history. Then load a new dataset.
- **Goal**: Verify chat history behavior across sessions.
- **Expected**: Chat history visible for current session. Behavior on dataset switch documented.

### T13.8 — Enter vs Shift+Enter
- **Action**: Type a message. Press Shift+Enter (new line). Then press Enter (send).
- **Goal**: Verify keyboard behavior.
- **Expected**: Shift+Enter adds newline in textarea. Enter sends the message.

---

## Module 14: Undo & Reset

### T14.1 — Single Undo
- **Action**: Load "moons". Select 5 points. Send: `these are one class`. Run clustering. Note the scatterplot state. Click "Undo".
- **Goal**: Verify single undo restores previous state.
- **Expected**: Scatterplot reverts to pre-constraint state. Clusters return to their previous assignments. Pending count and constraint history update.

### T14.2 — Multiple Sequential Undos
- **Action**: Apply 3 constraints (must_link, outlier_label, cluster_count), running clustering after each. Then click Undo 3 times.
- **Goal**: Verify chained undo (up to 10 steps).
- **Expected**: Each undo reverts to the previous state. After 3 undos, the scatterplot matches the initial cold-start clustering.

### T14.3 — Undo When No History
- **Action**: Load a fresh dataset. Click "Undo" immediately.
- **Goal**: Verify undo is disabled or shows error when no history.
- **Expected**: "Undo" button is disabled (grayed out). No crash.

### T14.4 — Reset
- **Action**: Load "blobs". Apply 3 constraints and run clustering. Click "Reset".
- **Goal**: Verify full reset clears all state.
- **Expected**: M reset to identity. DN and DO cleared. Clustering re-runs from cold start. Scatterplot shows initial clustering. Chat history may persist (verify behavior).

---

## Module 15: Export

### T15.1 — Export CSV After Clustering
- **Action**: Load "wine". Apply a must_link and an outlier_label constraint. Run clustering. Click "Export CSV".
- **Goal**: Verify exported file contents.
- **Expected**: Downloaded file (e.g., `wine_labeled.csv`) contains:
  - All 13 original feature columns (pre-normalization values).
  - `point_id` column.
  - `cluster_label` column (integer, -1 for outliers).
  - `is_outlier` column (boolean or 0/1).
  - If `include_scores=true`: `rscore`, `lscore`, `simscore`, `tscore` columns.
  - Correct number of rows (186).

### T15.2 — Export Before Clustering
- **Action**: Load "blobs" but do NOT run clustering. Click "Export CSV".
- **Goal**: Verify behavior when no clustering results exist.
- **Expected**: Either exports with default labels or shows an error/warning. No crash.

### T15.3 — Verify Outlier Labels in Export
- **Action**: Load "circles". Mark 5 points as outliers. Run. Export.
- **Goal**: Verify outlier flags in export.
- **Expected**: The 5 marked points have `is_outlier = true` and `cluster_label = -1` in the CSV.

---

## Module 16: Multiple Pending Constraints

### T16.1 — Queue Multiple Constraints Before Running
- **Action**: Load "blobs".
  1. Select 3 points → send: `these are one class` (must_link queued).
  2. Select 2 other points → send: `these are outliers` (outlier_label queued).
  3. Type: `split into 3 clusters` (cluster_count queued).
  4. Check: "Run clustering" button shows "(3 pending)".
  5. Click "Run clustering".
- **Goal**: Verify batch constraint application.
- **Expected**: All 3 constraints applied in one pipeline run. Scatterplot reflects all changes. Must-linked points same cluster, outlier points marked, ~3 clusters visible.

### T16.2 — Clear Pending Constraints
- **Action**: Queue 2 constraints (do NOT run). Click to clear pending (if UI supports) or send API call.
- **Goal**: Verify pending constraints can be discarded.
- **Expected**: Pending count returns to 0. "Run clustering" button no longer pulses.

---

## Module 17: Edge Cases & Error Handling

### T17.1 — Select All Points as Must-Link
- **Action**: Load "blobs" (212 points). Lasso-select ALL points. Send: `these are one class`. Run.
- **Goal**: Verify behavior when all points are must-linked.
- **Expected**: All points assigned to a single cluster. Legend shows 1 cluster.

### T17.2 — Mark All Points as Outliers
- **Action**: Load "blobs". Select all points. Send: `these are outliers`. Run.
- **Goal**: Verify edge case where all points are outliers.
- **Expected**: All points turn gray with dashed borders. 0 clusters, all outliers. Or system shows warning about this extreme case.

### T17.3 — Contradictory Constraints
- **Action**: Load "blobs".
  1. Select points [0,1,2]. Send: `these are one class`. Run.
  2. Stage Group A = [0,1], Group B = [2]. Send: `these should not be together`. Run.
- **Goal**: Verify behavior with contradictory must-link and cannot-link on overlapping points.
- **Expected**: System handles gracefully — later constraint may override earlier, or system shows a warning/conflict message. No crash.

### T17.4 — Cluster Count = 1
- **Action**: Type: `all points should be 1 cluster`. Run.
- **Goal**: Verify minimum cluster count.
- **Expected**: All points assigned to a single cluster.

### T17.5 — Cluster Count > Number of Points
- **Action**: Type: `split into 500 clusters`. Run.
- **Goal**: Verify handling of unreasonable cluster count.
- **Expected**: System either caps at a reasonable number, shows a warning, or creates many small clusters.

### T17.6 — Rapid Sequential Messages
- **Action**: Send 5 messages rapidly without waiting for responses.
- **Goal**: Verify chat handles rapid input without race conditions.
- **Expected**: All messages processed. Responses appear in correct order. No duplicate constraints.

### T17.7 — LLM Unavailable
- **Action**: Stop the Ollama server. Then send a message that requires LLM (not matching rule patterns).
- **Goal**: Verify graceful degradation when LLM is offline.
- **Expected**: Error message displayed in chat. Rule-based constraints should still work. Status indicator may turn red.

### T17.8 — Browser Refresh Mid-Session
- **Action**: Load "moons", apply constraints, run clustering. Press F5 to refresh browser.
- **Goal**: Verify session persistence after refresh.
- **Expected**: Depends on storage backend:
  - `memory`: Session lost, must start over.
  - `pickle`: Session restored, scatterplot shows previous state.

---

## Module 18: Status Indicator

### T18.1 — Status Transitions
- **Action**: Track the status dot color through a full workflow:
  1. Page load → Gray (Idle)
  2. Load dataset → briefly Orange (Working) → Green (Ready)
  3. Send chat message → briefly Orange (Working) → Green (Ready)
  4. Run clustering → Orange (Working) → Green (Ready)
- **Goal**: Verify correct status transitions.
- **Expected**: Colors match the documented states.

### T18.2 — Error Status
- **Action**: Trigger an error condition (e.g., LLM offline, invalid API call).
- **Goal**: Verify red error status.
- **Expected**: Status dot turns red. Error text displayed.

---

## Module 19: End-to-End Workflow

### T19.1 — Complete Wine Dataset Workflow
This is a full realistic workflow combining multiple features.

1. **Load**: Click "Load sample" → "wine". Verify: 186 points, 13 features.
2. **Initial observation**: Hover over points to see initial cluster assignments. Note the legend (e.g., 2 clusters + some outliers).
3. **Feature hint**: Type: `alcohol is not very important`. Follow up with `reduce it slightly`. Run clustering. Observe projection changes.
4. **Must-link**: Lasso 5 points that look like they should be grouped. Send: `these belong together`. Run. Verify they share a cluster.
5. **Outlier label**: Click 2 isolated points. Send: `these are anomalies`. Run. Verify gray + dashed styling.
6. **Cannot-link**: Stage two groups of 3 points each from different regions. Send: `these are different classes`. Run. Verify separation.
7. **Cluster count**: Type: `I want 4 clusters total`. Run. Verify 4 colors in legend.
8. **Triplet**: Stage 3 single points. Send: `the first is more like the second than the third`. Run. Observe metric change.
9. **Undo last step**: Click "Undo". Verify triplet effect reversed.
10. **Export**: Click "Export CSV". Open downloaded file. Verify all columns, correct labels, outlier flags.

### T19.2 — Complete Moons Dataset Workflow
1. **Load**: "moons" (320 points, 2 features).
2. **Observe**: Initial clustering likely merges parts of the two moons.
3. **Must-link**: Select points from one moon's tip. Send: `same class`. Run.
4. **Cannot-link**: Stage one group from each moon. Send: `different clusters`. Run.
5. **Observe**: The two moons should now be correctly separated into 2 clusters.
6. **Add noise as outliers**: Select scattered noise points between the moons. Send: `these are noise`. Run.
7. **Verify**: 2 clean clusters + outliers. Export and check.

---

## Module 20: Performance Observations

### T20.1 — Clustering Latency (Small Dataset)
- **Action**: Load "blobs" (212 points). Time the "Run clustering" operation.
- **Goal**: Baseline performance.
- **Expected**: Clustering completes in < 2 seconds.

### T20.2 — Clustering Latency (Medium Dataset)
- **Action**: Upload a ~1000-point CSV. Run clustering.
- **Goal**: Verify acceptable performance.
- **Expected**: Clustering completes in < 10 seconds.

### T20.3 — MDS Projection with Many Features
- **Action**: Upload a CSV with 50+ features, ~200 points. Run clustering.
- **Goal**: Verify MDS handles high-dimensional data.
- **Expected**: Projection computes successfully. May take longer than 2D datasets.

---

## Test Completion Checklist

| Module | Tests | Status |
|--------|-------|--------|
| M1: Startup & Health | T1.1–T1.3 | [ ] |
| M2: Data Loading | T2.1–T2.7 | [ ] |
| M3: Visualization | T3.1–T3.8 | [ ] |
| M4: Selection Groups | T4.1–T4.4 | [ ] |
| M5: Must-Link | T5.1–T5.4 | [ ] |
| M6: Cannot-Link | T6.1–T6.4 | [ ] |
| M7: Triplet | T7.1–T7.3 | [ ] |
| M8: Outlier Label | T8.1–T8.4 | [ ] |
| M9: Cluster Count | T9.1–T9.5 | [ ] |
| M10: Feature Hint | T10.1–T10.4 | [ ] |
| M11: Cluster Merge | T11.1–T11.3 | [ ] |
| M12: Reassign | T12.1–T12.3 | [ ] |
| M13: Chat & LLM | T13.1–T13.8 | [ ] |
| M14: Undo & Reset | T14.1–T14.4 | [ ] |
| M15: Export | T15.1–T15.3 | [ ] |
| M16: Multiple Pending | T16.1–T16.2 | [ ] |
| M17: Edge Cases | T17.1–T17.8 | [ ] |
| M18: Status Indicator | T18.1–T18.2 | [ ] |
| M19: End-to-End | T19.1–T19.2 | [ ] |
| M20: Performance | T20.1–T20.3 | [ ] |
| **Total** | **72 tests** | |
