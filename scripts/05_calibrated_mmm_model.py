import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

def geometric_adstock(x, alpha=0.5):
    """
    Applies geometric adstock decay transformation over a 1D time-series.
    """
    adstocked = np.zeros_like(x)
    for t in range(len(x)):
        if t == 0:
            adstocked[t] = x[t]
        else:
            adstocked[t] = x[t] + alpha * adstocked[t - 1]
    return adstocked

def run_calibrated_mmm(data_path="data/marketing_data.csv"):
    """
    Fits an Experimentally Calibrated Bayesian MMM in PyMC.
    Uses an informative Normal prior on TV ROAS (derived from Phase 4 DiD: 0.91x)
    and applies geometric adstock decay to model media memory carryover.
    """
    df = pd.read_csv(data_path)
    
    y = df['total_revenue'].to_numpy(dtype=float) / 10000.0
    x_meta = df['meta_spend'].to_numpy(dtype=float) / 10000.0
    x_google = df['google_spend'].to_numpy(dtype=float) / 10000.0
    x_tv = df['tv_spend'].to_numpy(dtype=float) / 10000.0
    x_promo = df['promo_flag'].to_numpy(dtype=float)
    
    # Pre-process TV spend with geometric decay (alpha = 0.5 carryover)
    x_tv_adstocked = geometric_adstock(x_tv, alpha=0.5)
    
    print("\nStarting PyMC Calibrated MCMC Sampling...")
    
    with pm.Model() as calibrated_model:
        # 1. Baseline & Uncalibrated Priors
        intercept = pm.Normal("baseline_intercept", mu=4.5, sigma=1.0)
        beta_meta = pm.HalfNormal("beta_meta", sigma=2.0)
        beta_google = pm.HalfNormal("beta_google", sigma=2.0)
        
        # 2. EXPERIMENTAL CALIBRATION:
        # Informative Gaussian prior anchored around Phase 4 DiD result (0.91x)
        beta_tv = pm.Normal("beta_tv", mu=0.91, sigma=0.10)
        
        beta_promo = pm.Normal("beta_promo", mu=1.5, sigma=0.5)
        sigma = pm.Exponential("sigma", lam=1.0)
        
        # 3. Linear Model Combination using Adstocked TV
        mu = (
            intercept 
            + (beta_meta * x_meta) 
            + (beta_google * x_google) 
            + (beta_tv * x_tv_adstocked) 
            + (beta_promo * x_promo)
        )
        
        # Likelihood
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y)
        
        # MCMC Sampling
        idata = pm.sample(
            draws=1000, 
            tune=1000, 
            chains=2, 
            return_inferencedata=True, 
            random_seed=42
        )
        
    print("\n--- MCMC Calibrated Diagnostics Summary ---")
    summary = az.summary(idata, var_names=["beta_meta", "beta_google", "beta_tv", "beta_promo", "baseline_intercept"])
    print(summary)
    
    # Visualizing Refined vs Original Posteriors
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    channels = [
        ("beta_meta", "Meta ROAS", 1.75), 
        ("beta_google", "Google ROAS", 2.30), 
        ("beta_tv", "TV ROAS (Calibrated via DiD)", 0.95)
    ]
    
    posterior_samples = idata.posterior
    
    for idx, (var_name, title, ref_val) in enumerate(channels):
        draws = posterior_samples[var_name].values.flatten()
        axes[idx].hist(draws, bins=30, density=True, alpha=0.6, color='#2ca02c', edgecolor='black')
        axes[idx].axvline(ref_val, color='red', linestyle='--', linewidth=2, label=f'Ground Truth ({ref_val})')
        axes[idx].axvline(np.mean(draws), color='darkgreen', linestyle='-', linewidth=2, label=f'Calibrated Mean ({np.mean(draws):.2f})')
        axes[idx].set_title(title, fontsize=11)
        axes[idx].legend()
        
    plt.suptitle("Phase 4 Calibrated PyMC Posterior ROAS Estimates", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig("data/calibrated_posterior_roas.png", dpi=300, bbox_inches='tight')
    print("\nCalibrated plot saved to 'data/calibrated_posterior_roas.png'")

if __name__ == "__main__":
    run_calibrated_mmm()