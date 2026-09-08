import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

def run_bayesian_forecasting(
    data_path="data/marketing_data.csv",
    inference_path="data/hill_mmm_inference_data.nc",
    forecast_weeks=12
):
    """
    Generates out-of-sample posterior predictive revenue forecasts with 89% 
    Bayesian prediction intervals under future media spend scenarios.
    """
    print("\n=======================================================")
    print("--- Phase 9: Out-of-Sample Bayesian Revenue Forecasting ---")
    print("=======================================================")
    
    # Load historical dataset and InferenceData
    df = pd.read_csv(data_path)
    idata = az.from_netcdf(inference_path)
    
    # Extract last known adstock states for forward propagation
    last_meta_spend = df['meta_spend'].iloc[-1] / 10000.0
    last_google_spend = df['google_spend'].iloc[-1] / 10000.0
    last_tv_spend = df['tv_spend'].iloc[-1] / 10000.0
    
    # Define Future Spend Scenario ($50,000 total weekly budget)
    # Scenario: Reallocated per SLSQP Optimizer (High Google, Moderate Meta, Low TV)
    future_meta_spend = np.full(forecast_weeks, 0.20)    # $2,000 / week
    future_google_spend = np.full(forecast_weeks, 4.70)  # $47,000 / week
    future_tv_spend = np.full(forecast_weeks, 0.10)      # $1,000 / week
    future_promo_flag = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]) # Projected promos
    
    # Propagate geometric adstocks into the future
    def project_adstock(future_spend, initial_val, alpha):
        adstocked = np.zeros(len(future_spend))
        for t in range(len(future_spend)):
            prev = initial_val if t == 0 else adstocked[t-1]
            adstocked[t] = future_spend[t] + alpha * prev
        return adstocked
        
    fut_meta_ad = project_adstock(future_meta_spend, last_meta_spend, alpha=0.3)
    fut_google_ad = project_adstock(future_google_spend, last_google_spend, alpha=0.2)
    fut_tv_ad = project_adstock(future_tv_spend, last_tv_spend, alpha=0.5)
    
    # Extract posterior chains
    post = idata.posterior
    intercepts = post["baseline_intercept"].values.flatten()
    beta_promos = post["beta_promo"].values.flatten()
    beta_metas = post["beta_meta"].values.flatten()
    beta_googles = post["beta_google"].values.flatten()
    beta_tvs = post["beta_tv"].values.flatten()
    
    K_metas, S_metas = post["K_meta"].values.flatten(), post["S_meta"].values.flatten()
    K_googles, S_googles = post["K_google"].values.flatten(), post["S_google"].values.flatten()
    K_tvs, S_tvs = post["K_tv"].values.flatten(), post["S_tv"].values.flatten()
    sigmas = post["sigma"].values.flatten()
    
    n_draws = len(intercepts)
    forecast_draws = np.zeros((n_draws, forecast_weeks))
    
    # Sample Posterior Predictive Distributions across draw parameter sets
    for i in range(n_draws):
        # Hill transformations for draw i
        h_meta = beta_metas[i] * (fut_meta_ad ** S_metas[i]) / (K_metas[i] ** S_metas[i] + fut_meta_ad ** S_metas[i])
        h_goog = beta_googles[i] * (fut_google_ad ** S_googles[i]) / (K_googles[i] ** S_googles[i] + fut_google_ad ** S_googles[i])
        h_tv = beta_tvs[i] * (fut_tv_ad ** S_tvs[i]) / (K_tvs[i] ** S_tvs[i] + fut_tv_ad ** S_tvs[i])
        
        mu_t = intercepts[i] + h_meta + h_goog + h_tv + (beta_promos[i] * future_promo_flag)
        # Add observation noise
        forecast_draws[i, :] = np.random.normal(mu_t, sigmas[i]) * 10000.0
        
    # Calculate Mean & 89% Prediction Intervals (HDIs)
    mean_forecast = np.mean(forecast_draws, axis=0)
    p_lower = np.percentile(forecast_draws, 5.5, axis=0)
    p_upper = np.percentile(forecast_draws, 94.5, axis=0)
    
    # Save Forecast Table
    future_dates = pd.date_range(start=pd.to_datetime(df['week_start'].iloc[-1]) + pd.Timedelta(weeks=1), periods=forecast_weeks, freq='W-MON')
    forecast_df = pd.DataFrame({
        "week_start": future_dates,
        "forecasted_revenue_mean": mean_forecast,
        "hdi_89_lower": p_lower,
        "hdi_89_upper": p_upper
    })
    
    forecast_df.to_csv("data/bayesian_12w_revenue_forecast.csv", index=False)
    print("\n12-Week Forecast Summary ($10,000 units converted):")
    print(forecast_df[["week_start", "forecasted_revenue_mean", "hdi_89_lower", "hdi_89_upper"]].to_string(index=False))
    print("\nForecast data saved to 'data/bayesian_12w_revenue_forecast.csv'")

if __name__ == "__main__":
    run_bayesian_forecasting()