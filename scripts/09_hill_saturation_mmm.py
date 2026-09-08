import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

def geometric_adstock_numpy(x, alpha=0.5):
    """Pre-processes 1D series with geometric adstock carryover."""
    adstocked = np.zeros_like(x)
    for t in range(len(x)):
        if t == 0:
            adstocked[t] = x[t]
        else:
            adstocked[t] = x[t] + alpha * adstocked[t - 1]
    return adstocked

def run_hill_saturation_mmm(data_path="data/marketing_data.csv"):
    """
    Fits a Bayesian MMM in PyMC incorporating explicit Hill Saturation Functions
    to capture channel-level diminishing marginal returns.
    """
    df = pd.read_csv(data_path)
    
    # Scale variables for numerical stability ($10,000 units)
    y = df['total_revenue'].to_numpy(dtype=float) / 10000.0
    x_meta = df['meta_spend'].to_numpy(dtype=float) / 10000.0
    x_google = df['google_spend'].to_numpy(dtype=float) / 10000.0
    x_tv = df['tv_spend'].to_numpy(dtype=float) / 10000.0
    x_promo = df['promo_flag'].to_numpy(dtype=float)
    
    # Apply geometric adstock carryover prior to model graph
    x_meta_ad = geometric_adstock_numpy(x_meta, alpha=0.3)
    x_google_ad = geometric_adstock_numpy(x_google, alpha=0.2)
    x_tv_ad = geometric_adstock_numpy(x_tv, alpha=0.5)
    
    print("\n--- Starting PyMC Sampling with Hill Saturation Transformations ---")
    
    with pm.Model() as hill_model:
        # 1. Baseline & Control Priors
        intercept = pm.Normal("baseline_intercept", mu=4.5, sigma=1.0)
        beta_promo = pm.Normal("beta_promo", mu=1.5, sigma=0.5)
        
        # 2. Maximum Channel Capacities (Max Beta Returns)
        beta_meta = pm.HalfNormal("beta_meta", sigma=3.0)
        beta_google = pm.HalfNormal("beta_google", sigma=4.0)
        
        # EXPERIMENTAL CALIBRATION PRIOR for TV Max Capacity
        beta_tv = pm.Normal("beta_tv", mu=1.2, sigma=0.2)
        
        # 3. Hill Saturation Parameters: K (Half-Saturation) & S (Slope)
        K_meta = pm.Gamma("K_meta", alpha=2.0, beta=2.0)
        K_google = pm.Gamma("K_google", alpha=2.0, beta=2.0)
        K_tv = pm.Gamma("K_tv", alpha=3.0, beta=1.5)
        
        S_meta = pm.Gamma("S_meta", alpha=3.0, beta=3.0)
        S_google = pm.Gamma("S_google", alpha=3.0, beta=3.0)
        S_tv = pm.Gamma("S_tv", alpha=3.0, beta=3.0)
        
        # 4. Hill Saturation Transformations
        hill_meta = beta_meta * (x_meta_ad ** S_meta) / (K_meta ** S_meta + x_meta_ad ** S_meta)
        hill_google = beta_google * (x_google_ad ** S_google) / (K_google ** S_google + x_google_ad ** S_google)
        hill_tv = beta_tv * (x_tv_ad ** S_tv) / (K_tv ** S_tv + x_tv_ad ** S_tv)
        
        # Deterministic nodes to track saturated channel contributions
        pm.Deterministic("contrib_meta", hill_meta)
        pm.Deterministic("contrib_google", hill_google)
        pm.Deterministic("contrib_tv", hill_tv)
        
        # 5. Expected Target Mean & Likelihood
        mu = intercept + hill_meta + hill_google + hill_tv + (beta_promo * x_promo)
        sigma = pm.Exponential("sigma", lam=1.0)
        
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)
        
        # MCMC Sampling
        idata = pm.sample(
            draws=1000, 
            tune=1000, 
            chains=2, 
            target_accept=0.92,
            random_seed=42
        )
        
    print("\n--- MCMC Hill Model Convergence Summary ---")
    # Clean print of summary table without fragile column filtering
    summary = az.summary(idata, var_names=["beta_meta", "beta_google", "beta_tv", "K_meta", "K_google", "K_tv"])
    print(summary)
    
    # Correct method call to write NetCDF file directly from InferenceData object
    os.makedirs("data", exist_ok=True)
    idata.to_netcdf("data/hill_mmm_inference_data.nc")
    print("\nInferenceData saved successfully to 'data/hill_mmm_inference_data.nc'")

if __name__ == "__main__":
    run_hill_saturation_mmm()