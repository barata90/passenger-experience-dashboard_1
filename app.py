import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

# -----------------------------
# Page config + light theming
# -----------------------------
st.set_page_config(
    page_title="Passenger Experience KPI Scorecard",
    layout="wide",
    page_icon="✈️"
)

st.markdown(
    """
    <style>
      /* Make metric cards look cleaner */
      [data-testid="stMetricValue"] { font-size: 28px; }
      [data-testid="stMetricDelta"] { font-size: 14px; }
      .small-note { color: rgba(255,255,255,0.65); font-size: 12px; }
      .insight-box {
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.03);
      }
    </style>
    """,
    unsafe_allow_html=True
)

BASE = Path(".")
OUTPUTS = BASE / "outputs"

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data():
    test_path = BASE / "test.csv"
    if not test_path.exists():
        st.error("test.csv not found in this folder.")
        st.stop()

    df = pd.read_csv(test_path)

    # basic cleanup aligned with notebook
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    for c in ["Customer Type", "Type of Travel"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title()

    if "Arrival Delay in Minutes" in df.columns:
        df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
            df["Arrival Delay in Minutes"].median()
        )

    # target flag
    df["is_satisfied"] = (
        df["satisfaction"].astype(str).str.lower().str.strip() == "satisfied"
    ).astype(int)

    # identify columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # touchpoints are numeric ratings excluding context/id/delay
    RATING_EXCLUDE = {
        "id", "Age", "Flight Distance",
        "Departure Delay in Minutes", "Arrival Delay in Minutes",
        "is_satisfied"
    }
    rating_cols = [c for c in numeric_cols if c not in RATING_EXCLUDE]

    # experience index
    df["experience_index"] = df[rating_cols].mean(axis=1)

    return df, rating_cols

@st.cache_data
def load_outputs():
    perm_path = OUTPUTS / "permutation_importance_all_features.csv"
    opp_path  = OUTPUTS / "touchpoint_opportunity_table.csv"
    imp_path  = OUTPUTS / "top3_opportunity_impact_simulation.csv"

    perm_df = pd.read_csv(perm_path) if perm_path.exists() else None
    opp_df  = pd.read_csv(opp_path)  if opp_path.exists()  else None
    imp_df  = pd.read_csv(imp_path)  if imp_path.exists()  else None
    return perm_df, opp_df, imp_df

df, rating_cols = load_data()
perm_df, opp_df, imp_df = load_outputs()

# -----------------------------
# Header
# -----------------------------
st.title("Case Study 1 — Passenger Experience KPI Scorecard")
st.caption("Product Development & Design Analytics • KPI Scorecard • Driver Modeling • Opportunity Prioritization")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Filters")
st.sidebar.caption("Filters apply to KPIs and all visuals below.")

# Reset button
if st.sidebar.button("Reset filters"):
    st.rerun()

# Filter defaults
all_class = sorted(df["Class"].dropna().unique().tolist()) if "Class" in df.columns else []
all_tot   = sorted(df["Type of Travel"].dropna().unique().tolist()) if "Type of Travel" in df.columns else []
all_cty   = sorted(df["Customer Type"].dropna().unique().tolist()) if "Customer Type" in df.columns else []

cls = st.sidebar.multiselect("Class", all_class, default=all_class)
tot = st.sidebar.multiselect("Type of Travel", all_tot, default=all_tot)
cty = st.sidebar.multiselect("Customer Type", all_cty, default=all_cty)

# Filtered dataframe
f = df.copy()
if "Class" in f.columns and cls:
    f = f[f["Class"].isin(cls)]
if "Type of Travel" in f.columns and tot:
    f = f[f["Type of Travel"].isin(tot)]
if "Customer Type" in f.columns and cty:
    f = f[f["Customer Type"].isin(cty)]

# Guard for empty filter results
if len(f) == 0:
    st.warning("No data matches the selected filters. Please adjust filters in the sidebar.")
    st.stop()

# -----------------------------
# Compute quick KPIs
# -----------------------------
overall_sat = df["is_satisfied"].mean() * 100
sat_rate = f["is_satisfied"].mean() * 100
delta_sat = sat_rate - overall_sat

exp_idx = f["experience_index"].mean()
overall_exp = df["experience_index"].mean()
delta_exp = exp_idx - overall_exp

n_rows = len(f)

wifi_low_share = None
if "Inflight wifi service" in f.columns:
    wifi_low_share = (f["Inflight wifi service"] <= 3).mean() * 100

# -----------------------------
# Tabs layout (WOW factor)
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Executive View", "Experience Deep Dive", "Model & Opportunities"])

# =========================================================
# TAB 1 — Executive View
# =========================================================
with tab1:
    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Satisfaction Rate", f"{sat_rate:.2f}%", f"{delta_sat:+.2f} pp vs overall")
    k2.metric("Avg Experience Index", f"{exp_idx:.3f}", f"{delta_exp:+.3f} vs overall")
    k3.metric("Passengers (Filtered)", f"{n_rows:,}")
    if wifi_low_share is not None:
        k4.metric("Wi-Fi ≤ 3 (Coverage)", f"{wifi_low_share:.2f}%")
    else:
        k4.metric("Wi-Fi ≤ 3 (Coverage)", "N/A")

    # Executive summary callouts
    st.markdown(
        f"""
        <div class="insight-box">
        <b>Executive Summary</b><br>
        • Satisfaction is <b>{sat_rate:.2f}%</b> under current filters (<b>{delta_sat:+.2f} pp</b> vs overall).<br>
        • Average Experience Index is <b>{exp_idx:.3f}</b> on a 0–5 scale (<b>{delta_exp:+.3f}</b> vs overall).<br>
        • Based on modeling, the highest-leverage initiatives are <b>Inflight Wi-Fi</b>, <b>Online Boarding</b>, and <b>Check-in Service</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Build touchpoint table for this filtered view (for opportunity map + bottom touchpoints)
    tp = []
    for c in rating_cols:
        avg_all = f[c].mean()
        avg_sat = f.loc[f["is_satisfied"] == 1, c].mean()
        avg_not = f.loc[f["is_satisfied"] == 0, c].mean()
        tp.append({
            "touchpoint": c,
            "avg_score_all": avg_all,
            "gap_sat_minus_not": avg_sat - avg_not
        })
    tp_df = pd.DataFrame(tp)

    # WOW chart: Opportunity map (performance vs impact)
    fig = px.scatter(
        tp_df,
        x="avg_score_all",
        y="gap_sat_minus_not",
        hover_name="touchpoint",
        title="Opportunity Map: Performance vs Impact (Touchpoints)",
        labels={
            "avg_score_all": "Avg touchpoint score (performance)",
            "gap_sat_minus_not": "Satisfaction gap (impact proxy)"
        }
    )
    # quadrant reference lines (medians)
    fig.add_vline(x=float(tp_df["avg_score_all"].median()))
    fig.add_hline(y=float(tp_df["gap_sat_minus_not"].median()))
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20), title_x=0.02)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Top 3 opportunities table (from notebook outputs), if available
    st.subheader("Top Product Opportunities (Model-based)")
    left, right = st.columns([1.2, 0.8])

    with left:
        if opp_df is not None:
            show_opp = opp_df.head(10).copy()
            st.dataframe(show_opp, use_container_width=True)
        else:
            st.info("touchpoint_opportunity_table.csv not found in outputs/. Run the notebook export to enable this table.")

    with right:
        if imp_df is not None:
            st.markdown("**What-if Impact Simulation (Top 3)**")
            st.dataframe(imp_df.sort_values("lift_pp", ascending=False), use_container_width=True)
        else:
            st.info("top3_opportunity_impact_simulation.csv not found in outputs/. Run the notebook export to enable lift estimates.")

    # Download button (portfolio-ready)
    st.download_button(
        "Download filtered data (CSV)",
        data=f.to_csv(index=False).encode("utf-8"),
        file_name="filtered_passenger_data.csv",
        mime="text/csv"
    )

# =========================================================
# TAB 2 — Experience Deep Dive
# =========================================================
with tab2:
    st.subheader("Touchpoint Performance (Average + Satisfaction Gap)")

    # recompute tp_df (kept local for clarity)
    tp = []
    for c in rating_cols:
        avg_all = f[c].mean()
        avg_sat = f.loc[f["is_satisfied"] == 1, c].mean()
        avg_not = f.loc[f["is_satisfied"] == 0, c].mean()
        tp.append({
            "touchpoint": c,
            "avg_score_all": avg_all,
            "gap_sat_minus_not": avg_sat - avg_not
        })
    tp_df = pd.DataFrame(tp)

    c1, c2 = st.columns(2)
    with c1:
        bottom = tp_df.sort_values("avg_score_all", ascending=True).head(7)
        fig = px.bar(
            bottom,
            x="avg_score_all",
            y="touchpoint",
            orientation="h",
            title="Bottom Touchpoints by Avg Score",
            labels={"avg_score_all": "Avg score (0–5)", "touchpoint": ""}
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20), title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        topgap = tp_df.sort_values("gap_sat_minus_not", ascending=False).head(10)
        fig = px.bar(
            topgap,
            x="gap_sat_minus_not",
            y="touchpoint",
            orientation="h",
            title="Top Touchpoints by Satisfaction Gap (Satisfied − Not)",
            labels={"gap_sat_minus_not": "Gap", "touchpoint": ""}
        )
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20), title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Segment KPIs")
    seg1, seg2, seg3 = st.columns(3)

    with seg1:
        if "Class" in f.columns:
            g = f.groupby("Class")["is_satisfied"].agg(["count", "mean"]).reset_index()
            g["satisfaction_rate_%"] = g["mean"] * 100
            fig = px.bar(
                g.sort_values("satisfaction_rate_%", ascending=False),
                x="Class",
                y="satisfaction_rate_%",
                title="Satisfaction Rate by Class",
                labels={"satisfaction_rate_%": "Satisfaction %"}
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20), title_x=0.02)
            st.plotly_chart(fig, use_container_width=True)

    with seg2:
        if "Type of Travel" in f.columns:
            g = f.groupby("Type of Travel")["is_satisfied"].agg(["count", "mean"]).reset_index()
            g["satisfaction_rate_%"] = g["mean"] * 100
            fig = px.bar(
                g.sort_values("satisfaction_rate_%", ascending=False),
                x="Type of Travel",
                y="satisfaction_rate_%",
                title="Satisfaction Rate by Type of Travel",
                labels={"satisfaction_rate_%": "Satisfaction %"}
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20), title_x=0.02)
            st.plotly_chart(fig, use_container_width=True)

    with seg3:
        if "Customer Type" in f.columns:
            g = f.groupby("Customer Type")["is_satisfied"].agg(["count", "mean"]).reset_index()
            g["satisfaction_rate_%"] = g["mean"] * 100
            fig = px.bar(
                g.sort_values("satisfaction_rate_%", ascending=False),
                x="Customer Type",
                y="satisfaction_rate_%",
                title="Satisfaction Rate by Customer Type",
                labels={"satisfaction_rate_%": "Satisfaction %"}
            )
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20), title_x=0.02)
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3 — Model & Opportunities
# =========================================================
with tab3:
    st.subheader("Drivers & Product Opportunities")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Top Drivers (Permutation Importance)**")
        if perm_df is not None:
            topn = perm_df.sort_values("importance_mean", ascending=False).head(15)
            fig = px.bar(
                topn.sort_values("importance_mean", ascending=True),
                x="importance_mean",
                y="feature",
                orientation="h",
                title="Top 15 Features by Permutation Importance (AUC drop)",
                labels={"importance_mean": "Mean AUC drop", "feature": ""}
            )
            fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20), title_x=0.02)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Higher value = larger performance drop when the feature is shuffled → stronger driver.")
        else:
            st.info("permutation_importance_all_features.csv not found in outputs/. Run the notebook export to enable this chart.")

    with colB:
        st.markdown("**Opportunities (Touchpoints)**")
        if opp_df is not None:
            show = opp_df.head(10).copy()
            st.dataframe(show, use_container_width=True)
            st.caption("Opportunity score = driver importance × score headroom (5 − avg touchpoint score).")
        else:
            st.info("touchpoint_opportunity_table.csv not found in outputs/. Run the notebook export to enable this table.")

    st.markdown("**Top 3 What-if Impact Simulation**")
    if imp_df is not None:
        st.dataframe(imp_df.sort_values("lift_pp", ascending=False), use_container_width=True)
    else:
        st.info("top3_opportunity_impact_simulation.csv not found in outputs/. Run the notebook export to enable lift estimates.")

    st.divider()

    # Extra: show raw tables for debugging (collapsible)
    with st.expander("Show raw data preview (debug)"):
        st.dataframe(f.head(25), use_container_width=True)
        st.caption("Preview of filtered records (first 25 rows).")

st.subheader("Initiative Tracker (Targets & Status)")

init_path = BASE / "initiatives.csv"
if init_path.exists():
    init_df = pd.read_csv(init_path)
    st.dataframe(init_df, use_container_width=True)

    # Simple KPI target progress (visual)
    fig = px.bar(
        init_df,
        x="initiative",
        y="target_pp",
        title="Target KPI Lift (pp) by Initiative",
        labels={"target_pp":"Target lift (pp)", "initiative":""}
    )
    fig.update_layout(height=420, xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Create initiatives.csv in the project folder to enable initiative tracking.")

