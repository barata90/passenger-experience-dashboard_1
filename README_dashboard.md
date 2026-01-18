# Passenger Experience KPI Scorecard (Streamlit Dashboard)

A portfolio-ready dashboard for **Passenger Experience analytics** (KPI scorecard + touchpoint deep-dive + model-driven opportunity prioritization) using the Airline Passenger Satisfaction dataset.

This project is designed to demonstrate skills aligned with **Senior Data Specialist | Product Development & Design**:
- KPI design & monitoring
- Touchpoint performance analysis (soft product + digital journey)
- Driver modeling and explainability (permutation importance)
- Translating insights into **prioritized product opportunities** + estimated KPI lift (what-if simulation)

---

## 1) Project Structure

Place these files in the **same project folder**:

```
Case Study 1 — Passenger Experience KPI Scorecard (Product Dev oriented)/
├─ app.py
├─ train.csv                      (optional for dashboard; used by notebook)
├─ test.csv                       (required for dashboard)
└─ outputs/                       (recommended; created by notebook export)
   ├─ kpi_summary.csv
   ├─ touchpoint_scores.csv
   ├─ segment_kpi_class.csv
   ├─ segment_kpi_type_of_travel.csv
   ├─ segment_kpi_customer_type.csv
   ├─ segment_kpi_gender.csv
   ├─ permutation_importance_all_features.csv
   ├─ touchpoint_opportunity_table.csv
   └─ top3_opportunity_impact_simulation.csv
```

> The dashboard can run with **only `test.csv`**.  
> To unlock the **Model & Opportunities** tab fully, generate the files under `outputs/` from the notebook.

---

## 2) Setup (Conda on macOS)

### 2.1 Create & activate environment
```bash
conda create -n qa-dashboard python=3.11 -y
conda activate qa-dashboard
```

### 2.2 Install dependencies
```bash
conda install -c conda-forge pandas numpy matplotlib plotly scikit-learn streamlit -y
```

Optional (only needed if you want to re-run the modeling notebook in the same env):
```bash
conda install -c conda-forge xgboost shap -y
```

---

## 3) Run the Dashboard

From the project folder:
```bash
cd "/Users/user/Desktop/Case Study 1 — Passenger Experience KPI Scorecard (Product Dev oriented)"
streamlit run app.py
```

Streamlit will open the app in your browser (or show a local URL in the terminal).

---

## 4) Data Expectations

### 4.1 Required file
- `test.csv` must exist in the project root (same folder as `app.py`).

### 4.2 Columns used
The dashboard expects at least these columns in `test.csv`:
- `satisfaction` (target label: `satisfied` vs `neutral or dissatisfied`)
- Segments: `Class`, `Type of Travel`, `Customer Type` (and optionally `Gender`)
- Touchpoint ratings (0–5), e.g.:
  - `Inflight wifi service`, `Online boarding`, `Checkin service`, `Seat comfort`, `Cleanliness`, etc.

### 4.3 Cleaning performed by the app
The app automatically:
- Drops `Unnamed: 0` (if present)
- Standardizes `Customer Type` and `Type of Travel` text formatting
- Imputes missing `Arrival Delay in Minutes` using the median
- Creates:
  - `is_satisfied` (1 if satisfied, else 0)
  - `experience_index` (mean of all touchpoint ratings per passenger)

---

## 5) How to Use the Dashboard

### Sidebar filters
You can filter the analysis by:
- **Class**
- **Type of Travel**
- **Customer Type**

All KPIs and charts update instantly based on selected filters.

---

## 6) Dashboard Tabs & Interpretation

### A) Executive View (Leadership-ready)
Purpose: provide a single-page story for decision-makers.

**KPI Cards**
- **Satisfaction Rate**: % satisfied in the filtered subset (delta vs overall shown in pp)
- **Avg Experience Index**: composite KPI on a 0–5 scale (delta vs overall)
- **Passengers (Filtered)**: sample size under current filters
- **Wi‑Fi ≤ 3 (Coverage)**: % of passengers with low Wi‑Fi rating (scope of improvement)

**Interpretation**
- If Satisfaction Rate improves in a segment (e.g., Business Class), it indicates **segment-based expectations/experience differences**.
- Experience Index is a useful **top-line product KPI**; it aligns strongly with satisfaction.

**Opportunity Map (Performance vs Impact)**
- X-axis: **Avg touchpoint score** (performance)
- Y-axis: **Satisfaction gap** = avg(satisfied) − avg(not satisfied) (impact proxy)

Quadrants:
- **Low score + high gap** → highest priority improvements  
- **High score + high gap** → protect strengths  
- **Low score + low gap** → investigate (may be low impact)  
- **High score + low gap** → lower priority  

**Model-based opportunities (if `outputs/` exists)**
- Ranked initiatives using `opportunity_score = importance × headroom`
- Top 3 includes a simple what-if lift estimate (percentage points)

---

### B) Experience Deep Dive (Analyst view)
Purpose: identify pain points and what differentiates satisfied vs not satisfied passengers.

**Bottom Touchpoints by Avg Score**
- Highlights lowest-performing experience areas.

**Top Touchpoints by Satisfaction Gap**
- Highlights touchpoints most associated with satisfaction differences.

**Segment KPIs**
- Satisfaction rate by **Class**, **Type of Travel**, and **Customer Type**.

---

### C) Model & Opportunities (Decision support)
Purpose: show explainability and prioritization using notebook outputs.

**Top Drivers (Permutation Importance)**
Permutation importance measures the **drop in AUC** when a feature is shuffled.
- Higher AUC drop = stronger driver.

**Opportunities Table**
- `score_headroom = 5 − avg_score_all`
- `opportunity_score = importance_mean × score_headroom`

**Top 3 What-if Impact Simulation**
Scenario:
- For each top touchpoint, increase rating by +1 **only for passengers with score ≤ 3** (cap at 5)
- Compare predicted satisfaction before vs after → **lift_pp (percentage points)**

> This is a **scenario estimate**, not a causal claim.

---

## 7) How to Regenerate the `outputs/` Tables

Run the notebook export step to generate:
- `permutation_importance_all_features.csv`
- `touchpoint_opportunity_table.csv`
- `top3_opportunity_impact_simulation.csv`
(and other KPI summary tables)

If these are missing, the dashboard will still run but will show info messages in the Model tab.

---

## 8) Portfolio Talking Points (Suggested Narrative)

Use this structure in interviews:
1) **Baseline KPI**: satisfaction rate ~44% (dataset-level)
2) **Experience Index**: satisfied passengers have a materially higher composite experience score
3) **Drivers**: Wi‑Fi + digital journey + travel context show strong influence in the model
4) **Opportunities**: prioritize by driver importance × performance headroom
5) **Quantified scenario**: what-if simulation suggests largest lift from Wi‑Fi and digital journey improvements
6) **Action plan**: define KPIs and validate via experiments (A/B tests / phased rollouts)

---

## 9) Known Limitations & Best Practices

- **Not causal**: model explanations and what-if simulations are associative, not causal.
- **Touchpoint ratings are self-reported** and can be closely related to satisfaction (inflating model performance).
- If `id` appears as important in modeling, consider dropping it for cleaner interpretation.
- Present results as:
  - “drivers in this dataset”
  - “scenario estimates”
  - “next steps: validate via experimentation”

---

## 10) Troubleshooting

**`test.csv not found`**
- Ensure `test.csv` is in the same folder as `app.py`.

**`outputs/*.csv not found`**
- Run notebook export, or confirm `outputs/` exists and contains the CSVs.

**Streamlit won’t start**
```bash
conda activate qa-dashboard
python -c "import streamlit; print('streamlit ok')"
```

**No data after filtering**
- Filters may remove all rows. Click **Reset filters** or widen selections.

---

## 11) Optional Enhancements
- Add drill-down views by segment (Business vs Personal travel)
- Add KPI targets (before/after) and initiative tracking
- Deploy to Streamlit Community Cloud for a shareable portfolio link
