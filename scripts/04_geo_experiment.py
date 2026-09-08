import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_geo_experiment():
    """
    Simulates a 12-week Geo-Experiment across Treatment (Region A) and Control (Region B).
    Applies Difference-in-Differences (DiD) to estimate true causal lift during a 4-week ad blackout.
    """
    np.random.seed(42)
    n_weeks = 12
    pre_period = 8  # Weeks 1 to 8: Pre-treatment baseline
    post_period = 4 # Weeks 9 to 12: Treatment period (Ad Blackout in Region A)
    
    weeks = np.arange(1, n_weeks + 1)
    
    # 1. Simulate Pre-Period Baseline Revenue for both regions (Parallel Trends)
    base_trend = 50000 + (weeks * 1200)
    
    # Control Region (Region B) - Business as Usual throughout
    control_rev = base_trend + np.random.normal(0, 1500, size=n_weeks)
    
    # Treatment Region (Region A) - Higher baseline traffic
    treatment_baseline = base_trend * 1.30 + np.random.normal(0, 1800, size=n_weeks)
    
    # True causal effect of media blackout in Region A (Weekly Revenue Drop)
    TRUE_CAUSAL_WEEKLY_LIFT = 18500.0
    
    treatment_rev = treatment_baseline.copy()
    # Apply media blackout penalty during post-period (weeks 9-12)
    treatment_rev[pre_period:] -= TRUE_CAUSAL_WEEKLY_LIFT
    
    # Assemble DataFrame
    df = pd.DataFrame({
        'week': weeks,
        'control_revenue': control_rev,
        'treatment_revenue': treatment_rev,
        'period': ['Pre' if w <= pre_period else 'Post' for w in weeks]
    })
    
    # 2. Difference-in-Differences (DiD) Calculation
    pre_control = df[df['period'] == 'Pre']['control_revenue'].mean()
    post_control = df[df['period'] == 'Post']['control_revenue'].mean()
    
    pre_treat = df[df['period'] == 'Pre']['treatment_revenue'].mean()
    post_treat = df[df['period'] == 'Post']['treatment_revenue'].mean()
    
    # DiD Estimator = (Post_Treat - Pre_Treat) - (Post_Control - Pre_Control)
    delta_treat = post_treat - pre_treat
    delta_control = post_control - pre_control
    did_causal_effect = delta_treat - delta_control
    
    # 3. Calculate Calibrated Empirical ROAS
    # Assuming $15,000/week was spent in Region A during normal operations
    weekly_spend_held_back = 15000.0
    estimated_experimental_roas = abs(did_causal_effect) / weekly_spend_held_back
    
    print("\n=======================================================")
    print("--- Phase 4: Geo-Experiment Difference-in-Differences ---")
    print("=======================================================")
    print(f"Pre-Period Control Mean Revenue:  ${pre_control:,.2f}")
    print(f"Post-Period Control Mean Revenue: ${post_control:,.2f} (Delta: ${delta_control:,.2f})")
    print(f"Pre-Period Treat Mean Revenue:    ${pre_treat:,.2f}")
    print(f"Post-Period Treat Mean Revenue:   ${post_treat:,.2f} (Delta: ${delta_treat:,.2f})")
    print("-------------------------------------------------------")
    print(f"Estimated DiD Causal Lift Effect: ${did_causal_effect:,.2f} per week")
    print(f"Experimental Calibrated ROAS:     {estimated_experimental_roas:.2f}x")
    print("=======================================================")
    
    # Save experiment summary data
    df.to_csv("data/geo_experiment_results.csv", index=False)
    
    # 4. Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(weeks, df['control_revenue'], marker='o', label='Control Region B (No Blackout)', color='blue')
    plt.plot(weeks, df['treatment_revenue'], marker='s', label='Treatment Region A (Blackout W9-12)', color='red')
    plt.axvline(x=8.5, color='black', linestyle='--', label='Experiment Starts (Week 9)')
    plt.title("Geo-Experiment Media Blackout: Parallel Trends & Causal Lift", fontsize=12)
    plt.xlabel("Week")
    plt.ylabel("Weekly Revenue ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/geo_experiment_did.png", dpi=300)
    print("Geo-experiment plot saved to 'data/geo_experiment_did.png'")
    
    return estimated_experimental_roas

if __name__ == "__main__":
    run_geo_experiment()