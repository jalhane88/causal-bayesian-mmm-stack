import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import arviz as az
import plotly.graph_objects as go

# Set page configuration first
st.set_page_config(
    page_title="Marketing Measurement & Scenario Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Resolve project root directory dynamically
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "marketing_data.csv"
NC_PATH = BASE_DIR / "data" / "hill_mmm_inference_data.nc"

st.title("📊 Marketing Measurement & Scenario Simulator")
st.markdown("---")

# Verify input files exist before processing
if not DATA_PATH.exists() or not NC_PATH.exists():
    st.error(f"❌ Missing required data files!")
    st.warning(f"Expected files:\n- `{DATA_PATH}`\n- `{NC_PATH}`")
    st.stop()

@st.cache_data
def load_data_and_posterior():
    try:
        df = pd.read_csv(DATA_PATH)
        idata = az.from_netcdf(NC_PATH)
        post = idata.posterior
        
        params = {
            "intercept": float(post["baseline_intercept"].mean()),
            "beta_promo": float(post["beta_promo"].mean()),
            "beta_meta": float(post["beta_meta"].mean()),
            "beta_google": float(post["beta_google"].mean()),
            "beta_tv": float(post["beta_tv"].mean()),
            "K_meta": float(post["K_meta"].mean()),
            "K_google": float(post["K_google"].mean()),
            "K_tv": float(post["K_tv"].mean()),
            "S_meta": float(post["S_meta"].mean()),
            "S_google": float(post["S_google"].mean()),
            "S_tv": float(post["S_tv"].mean())
        }
        return df, params
    except Exception as e:
        st.error(f"Failed to load dataset or NetCDF inference file: {e}")
        st.stop()

df, params = load_data_and_posterior()

# Sidebar: Interactive Budget Sliders
st.sidebar.header("🎯 Scenario Budget Allocation")
st.sidebar.markdown("Adjust weekly spend ($) to simulate revenue impact:")

spend_google = st.sidebar.slider("Google Search Spend ($)", 0, 60000, 47000, step=1000)
spend_meta = st.sidebar.slider("Meta Ads Spend ($)", 0, 60000, 2000, step=1000)
spend_tv = st.sidebar.slider("TV Advertising Spend ($)", 0, 60000, 1000, step=1000)
promo_active = st.sidebar.checkbox("Active Promotional Campaign", value=False)

total_spend = spend_google + spend_meta + spend_tv

# Helper function for Hill transformation
def hill_transform(x_spend, beta, K, S, alpha=0.3):
    x_scaled = x_spend / 10000.0
    x_ad = x_scaled / (1.0 - alpha) # steady-state adstock approximation
    response_scaled = beta * (x_ad ** S) / (K ** S + x_ad ** S)
    return response_scaled * 10000.0

# Calculate Saturated Channel Contributions
contrib_base = params["intercept"] * 10000.0
contrib_promo = (params["beta_promo"] * 10000.0) if promo_active else 0.0
contrib_google = hill_transform(spend_google, params["beta_google"], params["K_google"], params["S_google"], alpha=0.2)
contrib_meta = hill_transform(spend_meta, params["beta_meta"], params["K_meta"], params["S_meta"], alpha=0.3)
contrib_tv = hill_transform(spend_tv, params["beta_tv"], params["K_tv"], params["S_tv"], alpha=0.5)

projected_revenue = contrib_base + contrib_promo + contrib_google + contrib_meta + contrib_tv
blended_roas = (projected_revenue - contrib_base) / total_spend if total_spend > 0 else 0.0

# Executive Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Weekly Spend", f"${total_spend:,.0f}")
col2.metric("Projected Weekly Revenue", f"${projected_revenue:,.0f}")
col3.metric("Baseline Revenue (Organic)", f"${contrib_base:,.0f}")
col4.metric("Blended Incremental ROAS", f"{blended_roas:.2f}x")

st.markdown("---")

# Revenue Waterfall Breakdown
st.subheader("💡 Predicted Revenue Breakdown by Channel Contribution")

fig_waterfall = go.Figure(go.Waterfall(
    name="Revenue", orientation="v",
    measure=["relative", "relative", "relative", "relative", "relative", "total"],
    x=["Baseline (Organic)", "Promo Lift", "Google Search", "Meta Ads", "TV Advertising", "Total Revenue"],
    textposition="outside",
    text=[f"${contrib_base:,.0f}", f"${contrib_promo:,.0f}", f"${contrib_google:,.0f}", 
          f"${contrib_meta:,.0f}", f"${contrib_tv:,.0f}", f"${projected_revenue:,.0f}"],
    y=[contrib_base, contrib_promo, contrib_google, contrib_meta, contrib_tv, projected_revenue],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
))

fig_waterfall.update_layout(
    title="Simulated Weekly Revenue Components",
    yaxis_title="Revenue ($)",
    waterfallgap=0.3,
    height=480
)

st.plotly_chart(fig_waterfall, width='stretch')