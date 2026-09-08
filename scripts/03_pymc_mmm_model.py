import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

def fit_bayesian_mmm(data_path="data/marketing_data.csv"):
    """
    Fits an initial observational Bayesian Marketing Mix Model using PyMC.
    Estimates posterior distributions for Meta, Google, and TV ROAS parameters.
    """
    df = pd.read_csv(data_path)
    
    # Extract features and target variable cleanly (prevents type-checker warnings)
    y = df['total_revenue'].to_numpy(dtype=float)
    x_meta = df['meta_spend'].to_numpy(dtype=float)
    x_google = df['google_spend'].to_numpy(dtype=float)
    x_tv = df['tv_spend'].to_numpy(dtype=float)
    x_promo = df['promo_flag'].to_numpy(dtype=float)
    
    # Scale features by $10,000 for numeric stability during MCMC sampling
    y_scaled = y / 10000.0
    meta_scaled = x_meta / 10000.0
    google_scaled = x_google / 10000.0
    tv_scaled = x_tv / 10000.0
    
    print("\nStarting PyMC Bayesian MCMC Sampling...")
    
    with pm.Model() as mmm_model:
        # Priors
        intercept = pm.Normal("baseline_intercept", mu=4.5, sigma=1.0)
        
        # Non-negative priors for channel ROAS (media impact cannot be negative)
        beta_meta = pm.HalfNormal("beta_meta", sigma=2.0)
        beta_google = pm.HalfNormal("beta_google", sigma=2.0)
        beta_tv = pm.HalfNormal("beta_tv", sigma=2.0)
        
        # Exogenous promotional coefficient
        beta_promo = pm.Normal("beta_promo", mu=1.5, sigma=0.5)
        
        # Error term
        sigma = pm.Exponential("sigma", lam=1.0)
        
        # Expected Revenue Linear Combination
        mu = (
            intercept 
            + (beta_meta * meta_scaled) 
            + (beta_google * google_scaled) 
            + (beta_tv * tv_scaled) 
            + (beta_promo * x_promo)
        )
        
        # Likelihood specification
        likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y_scaled)
        
        # MCMC Sampling (2 chains, 1000 draws each)
        idata = pm.sample(
            draws=1000, 
            tune=1000, 
            chains=2, 
            return_inferencedata=True, 
            random_seed=42
        )
    
    # Print Diagnostics & Summary
    print("\n--- MCMC Posterior Diagnostics Summary ---")
    summary = az.summary(idata, var_names=["beta_meta", "beta_google", "beta_tv", "beta_promo", "baseline_intercept"])
    print(summary)
        
    # Plotting Posteriors with Matplotlib directly
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    channels = [("beta_meta", "Meta ROAS", 1.75), 
                ("beta_google", "Google ROAS", 2.30), 
                ("beta_tv", "TV ROAS", 0.95)]
    
    posterior_samples = idata.posterior
    
    for idx, (var_name, title, ref_val) in enumerate(channels):
        draws = posterior_samples[var_name].values.flatten()
        axes[idx].hist(draws, bins=30, density=True, alpha=0.6, color='#1f77b4', edgecolor='black')
        axes[idx].axvline(ref_val, color='red', linestyle='--', linewidth=2, label=f'Ground Truth ({ref_val})')
        axes[idx].axvline(np.mean(draws), color='green', linestyle='-', linewidth=2, label=f'Posterior Mean ({np.mean(draws):.2f})')
        axes[idx].set_title(title, fontsize=12)
        axes[idx].legend()
        
    plt.suptitle("PyMC Posterior ROAS Estimates vs Ground Truth References", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig("data/posterior_roas_estimates.png", dpi=300, bbox_inches='tight')
    print("\nPosterior plot successfully saved to 'data/posterior_roas_estimates.png'")

if __name__ == "__main__":
    fit_bayesian_mmm()