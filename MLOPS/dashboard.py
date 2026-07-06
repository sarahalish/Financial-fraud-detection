"""Interactive AML Risk Dashboard (Streamlit).

Reads the scored output CSVs produced by the pipeline — no re-training,
no model code, just exploration of saved results.

Run from the project root:
    streamlit run dashboard.py
Then the browser opens automatically at http://localhost:8501
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AML Risk Dashboard", page_icon="🛡️", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
SCORED_CANDIDATES = [
    PROJECT_ROOT / "reports" / "scored_new_data.csv",
    PROJECT_ROOT / "data" / "processed" / "scored_training_data.csv",
]
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


# ----------------------------- data loading -----------------------------
@st.cache_data
def load_scored(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


available = [p for p in SCORED_CANDIDATES if p.exists()]
if not available:
    st.error(
        "No scored data found. Run `python run_pipeline.py` (and optionally "
        "`python predict.py ...`) first, then reload this page."
    )
    st.stop()

st.title("🛡️ AML/KYC Fraud Risk Dashboard")

source = st.sidebar.selectbox(
    "Data source", available, format_func=lambda p: p.relative_to(PROJECT_ROOT).as_posix()
)
df = load_scored(str(source))

# ----------------------------- sidebar filters -----------------------------
st.sidebar.header("Filters")

min_risk = st.sidebar.slider("Minimum final fraud risk", 0.0, 1.0, 0.0, 0.01)

alerts_only = st.sidebar.checkbox("Alerts only (is_fraud_alert = 1)", value=False)

peer_dev_min = 0.0
if "peer_deviation" in df.columns:
    peer_dev_min = st.sidebar.slider(
        "Minimum peer deviation (× sector average)", 0.0, 10.0, 0.0, 0.5
    )

sectors = sorted(df["sector"].dropna().unique()) if "sector" in df.columns else []
sel_sectors = st.sidebar.multiselect("Sectors", sectors, default=[])

client_query = st.sidebar.text_input("Client ID (exact match, optional)")

# apply filters
view = df[df["final_fraud_risk"] >= min_risk]
if alerts_only and "is_fraud_alert" in view.columns:
    view = view[view["is_fraud_alert"] == 1]
if peer_dev_min > 0 and "peer_deviation" in view.columns:
    view = view[view["peer_deviation"] >= peer_dev_min]
if sel_sectors:
    view = view[view["sector"].isin(sel_sectors)]
if client_query.strip():
    try:
        view = view[view["client_id"] == int(client_query)]
    except ValueError:
        st.sidebar.warning("Client ID must be a number.")

# ----------------------------- KPI row -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions (filtered)", f"{len(view):,}")
if "is_fraud_alert" in view.columns:
    n_alerts = int(view["is_fraud_alert"].sum())
    c2.metric("Alerts", f"{n_alerts:,}",
              f"{100 * n_alerts / max(len(view), 1):.1f}% of filtered")
c3.metric("Avg final risk", f"{view['final_fraud_risk'].mean():.3f}" if len(view) else "—")
c4.metric("Max final risk", f"{view['final_fraud_risk'].max():.3f}" if len(view) else "—")

# training metrics, if present
if METRICS_PATH.exists():
    with st.expander("Last training run — evaluation metrics (proxy labels)"):
        st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))

st.divider()

# ----------------------------- charts -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Risk score distribution")
    color_col = "is_fraud_alert" if "is_fraud_alert" in view.columns else None
    fig = px.histogram(
        view, x="final_fraud_risk", nbins=60, color=color_col,
        color_discrete_map={0: "#2e7d32", 1: "#c62828"},
        labels={"final_fraud_risk": "Final fraud risk", "is_fraud_alert": "Alert"},
    )
    fig.update_layout(height=380, bargap=0.02, legend_title_text="Alert")
    st.plotly_chart(fig, width="stretch")

with right:
    if "sector" in view.columns and "is_fraud_alert" in view.columns and len(view):
        st.subheader("Alerts by sector")
        by_sector = (
            view.groupby("sector")["is_fraud_alert"].sum()
            .sort_values(ascending=True).reset_index()
        )
        fig2 = px.bar(
            by_sector, x="is_fraud_alert", y="sector", orientation="h",
            labels={"is_fraud_alert": "Number of alerts", "sector": ""},
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, width="stretch")

# amount vs risk scatter (sampled so it stays snappy)
if "amount" in view.columns and len(view):
    st.subheader("Amount vs. risk")
    sample = view.sample(min(len(view), 5000), random_state=0)
    fig3 = px.scatter(
        sample, x="amount", y="final_fraud_risk",
        color="is_fraud_alert" if "is_fraud_alert" in sample.columns else None,
        color_discrete_map={0: "#2e7d32", 1: "#c62828"},
        hover_data=[c for c in ["transaction_id", "client_id", "sector",
                                "peer_deviation"] if c in sample.columns],
        log_x=True, opacity=0.5,
        labels={"amount": "Amount (log scale)", "final_fraud_risk": "Final fraud risk",
                "is_fraud_alert": "Alert"},
    )
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, width="stretch")

st.divider()

# ----------------------------- alert queue table -----------------------------
st.subheader("Priority queue")
sort_col = "refined_priority" if "refined_priority" in view.columns else "final_fraud_risk"
show_cols = [c for c in [
    "transaction_id", "client_id", "client_name", "amount", "sector",
    "transaction_type", "final_fraud_risk", "behavioral_risk_score",
    "static_risk_score", "peer_deviation", "refined_priority", "is_fraud_alert",
] if c in view.columns]

top_n = st.slider("Rows to display", 10, 500, 50, 10)
table = view.sort_values(sort_col, ascending=False)[show_cols].head(top_n)
st.dataframe(
    table.style.background_gradient(subset=["final_fraud_risk"], cmap="Reds"),
    width="stretch", height=450,
)

# download the current filtered view
st.download_button(
    "⬇️ Download filtered results as CSV",
    view[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="filtered_aml_results.csv",
    mime="text/csv",
)
