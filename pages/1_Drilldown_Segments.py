import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Drilldown — Segments", layout="wide")

BASE = Path(".")
test_path = BASE / "test.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(test_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    for c in ["Customer Type", "Type of Travel"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.title()
    if "Arrival Delay in Minutes" in df.columns:
        df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Arrival Delay in Minutes"].median())
    df["is_satisfied"] = (df["satisfaction"].astype(str).str.lower().str.strip() == "satisfied").astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    RATING_EXCLUDE = {"id","Age","Flight Distance","Departure Delay in Minutes","Arrival Delay in Minutes","is_satisfied"}
    rating_cols = [c for c in numeric_cols if c not in RATING_EXCLUDE]
    df["experience_index"] = df[rating_cols].mean(axis=1)
    return df, rating_cols

df, rating_cols = load_data()

st.title("Drilldown — Segment Deep Dive")
st.caption("Compare Business vs Personal travel, and drill into touchpoints & satisfaction gaps.")

# Segment selector
seg = st.selectbox("Type of Travel", sorted(df["Type of Travel"].dropna().unique()))
d = df[df["Type of Travel"] == seg].copy()

# KPI row
c1, c2, c3 = st.columns(3)
c1.metric("Satisfaction Rate", f"{d['is_satisfied'].mean()*100:.2f}%")
c2.metric("Avg Experience Index", f"{d['experience_index'].mean():.3f}")
c3.metric("Passengers", f"{len(d):,}")

st.divider()

# Touchpoint gap within this segment
tp = []
for c in rating_cols:
    avg_all = d[c].mean()
    avg_sat = d.loc[d["is_satisfied"]==1, c].mean()
    avg_not = d.loc[d["is_satisfied"]==0, c].mean()
    tp.append({"touchpoint": c, "avg_score": avg_all, "gap_sat_minus_not": avg_sat-avg_not})

tp_df = pd.DataFrame(tp)

left, right = st.columns(2)
with left:
    bottom = tp_df.sort_values("avg_score").head(7)
    fig = px.bar(bottom, x="avg_score", y="touchpoint", orientation="h",
                 title=f"Bottom Touchpoints — {seg}")
    st.plotly_chart(fig, use_container_width=True)

with right:
    topgap = tp_df.sort_values("gap_sat_minus_not", ascending=False).head(10)
    fig = px.bar(topgap, x="gap_sat_minus_not", y="touchpoint", orientation="h",
                 title=f"Top Satisfaction Gaps — {seg}")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Segment drill by Class
if "Class" in d.columns:
    g = d.groupby("Class")["is_satisfied"].mean().reset_index()
    g["satisfaction_rate_%"] = g["is_satisfied"]*100
    fig = px.bar(g.sort_values("satisfaction_rate_%", ascending=False),
                 x="Class", y="satisfaction_rate_%", title=f"Satisfaction by Class — {seg}")
    st.plotly_chart(fig, use_container_width=True)
