-- sql/cohort_analytics.sql
-- Consultancy SQL Engine: Multi-Channel Efficiency, Rolling Performance, and Intent Metrics

WITH raw_marketing AS (
    SELECT 
        CAST(week_start AS DATE) AS week_start,
        total_revenue,
        meta_spend,
        google_spend,
        tv_spend,
        promo_flag,
        share_of_search_index,
        invalid_traffic_pct,
        (meta_spend + google_spend + tv_spend) AS total_paid_spend
    FROM 'data/marketing_data.csv'
),

metrics_calculated AS (
    SELECT
        week_start,
        total_revenue,
        total_paid_spend,
        meta_spend,
        google_spend,
        tv_spend,
        promo_flag,
        share_of_search_index,
        invalid_traffic_pct,
        
        -- Traffic Quality Adjustment: Isolating Valid Revenue/Spend Impact
        ROUND(total_paid_spend * (1 - invalid_traffic_pct), 2) AS effective_paid_spend,
        
        -- Blended Spend Efficiency Ratio (Blended ROAS)
        ROUND(total_revenue / NULLIF(total_paid_spend, 0), 2) AS blended_roas,
        
        -- Net Revenue after Paid Media Costs
        ROUND(total_revenue - total_paid_spend, 2) AS net_margin
    FROM raw_marketing
)

SELECT
    week_start,
    total_revenue,
    total_paid_spend,
    effective_paid_spend,
    blended_roas,
    share_of_search_index,
    invalid_traffic_pct,
    
    -- 4-Week Moving Average of Revenue (Smooth Out Weekly Noise)
    ROUND(AVG(total_revenue) OVER (
        ORDER BY week_start 
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_4wk_avg_revenue,
    
    -- 4-Week Moving Average of Blended ROAS
    ROUND(AVG(blended_roas) OVER (
        ORDER BY week_start 
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ), 2) AS rolling_4wk_avg_roas,
    
    -- Year-over-Year Revenue Lag Comparison (52 Weeks Prior)
    LAG(total_revenue, 52) OVER (ORDER BY week_start) AS rev_52_weeks_ago,
    
    -- Exogenous Promo Flag
    promo_flag

FROM metrics_calculated
ORDER BY week_start ASC;