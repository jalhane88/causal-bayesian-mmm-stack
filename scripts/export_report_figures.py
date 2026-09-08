# scripts/export_report_figures.py
from pathlib import Path
import arviz as az
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

idata = az.from_netcdf(BASE_DIR / "data" / "hill_mmm_inference_data.nc")

# 1. Export Trace Plots
az.plot_trace(idata, var_names=["beta_meta", "beta_google", "beta_tv"])
plt.savefig(FIGURES_DIR / "01_mcmc_trace_convergence.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Export Forest Plot
az.plot_forest(idata, var_names=["beta_meta", "beta_google", "beta_tv"], combined=True, ci_probs=(0.50, 0.89))
plt.savefig(FIGURES_DIR / "02_posterior_capacities_forest.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Saved diagnostic figures to {FIGURES_DIR}")