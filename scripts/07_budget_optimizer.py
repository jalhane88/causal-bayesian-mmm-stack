import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_media_budget(total_budget=50000.0):
    """
    Uses PyMC posterior ROAS estimates to calculate the optimal 
    media budget allocation across Meta, Google, and TV using SciPy.
    """
    # 1. Posterior Means from Calibrated MMM (Phase 4/5)
    beta_meta = 1.83
    beta_google = 2.94
    beta_tv = 0.91
    
    # Objective Function: Maximize Revenue -> Minimize (-1 * Estimated Revenue)
    def objective(spend):
        s_meta, s_google, s_tv = spend
        # Diminishing returns approximation via square-root elasticity
        rev_meta = beta_meta * (s_meta ** 0.85)
        rev_google = beta_google * (s_google ** 0.85)
        rev_tv = beta_tv * (s_tv ** 0.85)
        
        total_revenue = rev_meta + rev_google + rev_tv
        return -total_revenue  # Negated for minimization
    
    # Constraints & Bounds
    # Total spend must equal the target weekly budget
    constraints = ({'type': 'eq', 'fun': lambda spend: total_budget - sum(spend)})
    
    # Minimum and maximum channel spend bounds (prevent 0 allocation)
    bounds = (
        (2000.0, total_budget),  # Meta min $2k
        (2000.0, total_budget),  # Google min $2k
        (1000.0, total_budget)   # TV min $1k
    )
    
    # Initial Guess: Equal allocation
    initial_spend = [total_budget / 3.0] * 3
    
    # Optimize
    res = minimize(objective, initial_spend, method='SLSQP', bounds=bounds, constraints=constraints)
    
    opt_meta, opt_google, opt_tv = res.x
    max_predicted_revenue = -res.fun
    
    print("\n=======================================================")
    print(f"--- Phase 6: Bayesian Budget Optimization (${total_budget:,.0f} Budget) ---")
    print("=======================================================")
    print(f"Optimized Meta Spend:   ${opt_meta:,.2f} ({opt_meta/total_budget:.1%})")
    print(f"Optimized Google Spend: ${opt_google:,.2f} ({opt_google/total_budget:.1%})")
    print(f"Optimized TV Spend:     ${opt_tv:,.2f} ({opt_tv/total_budget:.1%})")
    print("-------------------------------------------------------")
    print(f"Predicted Incremental Revenue: ${max_predicted_revenue:,.2f}")
    print("=======================================================")
    
    # Save optimized strategy summary
    summary_df = pd.DataFrame([{
        "total_budget": total_budget,
        "meta_allocation": opt_meta,
        "google_allocation": opt_google,
        "tv_allocation": opt_tv,
        "predicted_revenue": max_predicted_revenue
    }])
    summary_df.to_csv("data/optimized_budget_allocation.csv", index=False)
    print("Optimization results saved to 'data/optimized_budget_allocation.csv'")

if __name__ == "__main__":
    optimize_media_budget(total_budget=50000.0)