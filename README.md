## Core Methodology Pillars

1. **SQL Transformation Layer (`scripts/02_duckdb_pipeline.py`):** Ingests raw time-series marketing data using DuckDB CTEs and windowing functions for robust feature engineering.
2. **Observational Bayesian MMM (`scripts/03_pymc_mmm_model.py`):** Uses PyMC to estimate channel elasticities under non-negative HalfNormal priors.
3. **Quasi-Experimental Calibration (`scripts/04_geo_experiment.py`):** Runs a Difference-in-Differences (DiD) quasi-experiment across treatment/control regions during a 4-week blackout period to isolate TV's true causal return.
4. **Calibrated MMM Refinement (`scripts/05_calibrated_mmm_model.py`):** Refines PyMC priors using experimental findings and incorporates geometric adstock carryover decay.
5. **Unstructured Data Parsing (`scripts/06_claude_survey_classifier.py`):** Uses Claude 3.5 Sonnet to categorize raw customer survey text into attribution channels.
6. **Bayesian Budget Optimization (`scripts/07_budget_optimizer.py`):** Solves a constrained nonlinear optimization problem (SLSQP) to maximize predicted revenue.

## Quickstart

```bash
# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline
python scripts/01_generate_data.py
python scripts/02_duckdb_pipeline.py
python scripts/03_pymc_mmm_model.py
python scripts/04_geo_experiment.py
python scripts/05_calibrated_mmm_model.py
python scripts/06_claude_survey_classifier.py
python scripts/07_budget_optimizer.py
python scripts/08_generate_report.py
python scripts/09_hill_saturation_mmm.py
python scripts/10_bayesian_forecasting.py
python scripts/11_streamlit_dashboard.py
python scripts 12_causal_synthetic_control_calibration.py


marketing-analytics-stack/          <-- Root Project Directory
│
├── main.py                         <-- CLI Orchestrator File (CREATED IN ROOT)
├── README.md                       <-- Project Documentation & Setup Guide
├── requirements.txt                <-- Environment Dependencies
│
├── data/                           <-- Data Storage Layer
│   ├── marketing_data.csv          <-- Synthetic / Raw Channel Spend & Revenue
│   └── hill_mmm_inference_data.nc  <-- NetCDF4 Bayesian Posterior Storage
│
├── scripts/                        <-- Production Python Scripts (Execution Engines)
│   ├── 11_streamlit_dashboard.py   <-- Streamlit Scenario Planner UI & Waterfall
│   └── 12_causal_synthetic_control_calibration.py <-- SCM Geo-Test Calibration & PyMC MMM
│
└── notebooks/                      <-- Diagnostic & Reporting Suites
    ├── 01_eda_and_vif_diagnostics.ipynb
    ├── 02_hill_saturation_response_curves.ipynb
    ├── 03_bayesian_forecasting_viz.ipynb
    └── 04_model_diagnostics.ipynb  <-- ArViZ Trace Plots, Energy, & Forest Plots