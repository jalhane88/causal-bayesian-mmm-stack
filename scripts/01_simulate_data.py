import numpy as np
import pandas as pd
import os

def generate_consultancy_dataset(n_weeks=130, seed=42):
    """
    Generates realistic marketing time-series data with adstock carryover,
    diminishing returns, organic seasonality, and exogenous confounders.
    """
    np.random.seed(seed)
    
    # 1. Timeline: 2.5 years of weekly data
    dates = pd.date_range(start="2024-01-01", periods=n_weeks, freq="W-MON")
    
    # 2. Organic Baseline & Seasonality
    time_trend = np.linspace(0, 1.5, n_weeks)
    seasonality = np.sin(2 * np.pi * dates.dayofyear / 365.25) * 12000
    baseline_revenue = 45000 + (time_trend * 3000) + seasonality
    
    # 3. Simulate Paid Media Spends ($)
    meta_spend = np.random.uniform(4000, 15000, size=n_weeks)
    google_spend = np.random.uniform(3000, 12000, size=n_weeks)
    
    # TV spend is bursty (zeros most weeks, big spends occasionally)
    tv_raw = np.random.choice([0, 1], size=n_weeks, p=[0.7, 0.3])
    tv_spend = tv_raw * np.random.uniform(20000, 50000, size=n_weeks)
    
    # 4. Apply Geometric Adstock Decay (Memory Effect)
    def apply_adstock(spend, alpha):
        adstocked = np.zeros_like(spend)
        for t in range(len(spend)):
            if t == 0:
                adstocked[t] = spend[t]
            else:
                adstocked[t] = spend[t] + alpha * adstocked[t - 1]
        return adstocked

    meta_adstocked = apply_adstock(meta_spend, alpha=0.3)
    google_adstocked = apply_adstock(google_spend, alpha=0.1)
    tv_adstocked = apply_adstock(tv_spend, alpha=0.6) # High memory decay
    
    # 5. Ground Truth Parameters (Incremental ROAS Multipliers)
    TRUE_META_ROAS = 1.75
    TRUE_GOOGLE_ROAS = 2.30
    TRUE_TV_ROAS = 0.95  # Upper-funnel brand, lower direct ROAS
    
    # Media Incremental Revenue Contribution
    meta_rev = meta_adstocked * TRUE_META_ROAS
    google_rev = google_adstocked * TRUE_GOOGLE_ROAS
    tv_rev = (tv_adstocked ** 0.85) * (TRUE_TV_ROAS * 10) # Log/Power saturation
    
    # 6. Confounders & Signals
    # Promotional Flag (Flash sales drive baseline up by 25%)
    promo_flag = np.random.choice([0, 1], size=n_weeks, p=[0.85, 0.15])
    promo_impact = promo_flag * 15000
    
    # Branded Share of Search Index (0 to 100 scale)
    share_of_search = 50 + (seasonality / 1000) + (tv_adstocked / 2000) + np.random.normal(0, 3, n_weeks)
    
    # Mid-Funnel Metric: Invalid Traffic Rate (e.g., 2% to 12% junk traffic)
    invalid_traffic_pct = np.random.uniform(0.02, 0.12, size=n_weeks)
    
    # 7. Total Revenue Combination + Random Noise
    noise = np.random.normal(0, 3500, size=n_weeks)
    total_revenue = baseline_revenue + meta_rev + google_rev + tv_rev + promo_impact + noise
    
    # 8. Assemble Master DataFrame
    df = pd.DataFrame({
        'week_start': dates,
        'total_revenue': np.round(total_revenue, 2),
        'meta_spend': np.round(meta_spend, 2),
        'google_spend': np.round(google_spend, 2),
        'tv_spend': np.round(tv_spend, 2),
        'promo_flag': promo_flag,
        'share_of_search_index': np.round(share_of_search, 2),
        'invalid_traffic_pct': np.round(invalid_traffic_pct, 4)
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/marketing_data.csv', index=False)
    print("Dataset generated successfully at 'data/marketing_data.csv'")
    print(f"Total Rows: {len(df)} | Revenue Mean: ${df['total_revenue'].mean():,.2f}")
    return df

if __name__ == "__main__":
    generate_consultancy_dataset()