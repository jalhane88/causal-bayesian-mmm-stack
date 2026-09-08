import duckdb
import pandas as pd

def run_sql_pipeline():
    """
    Executes the analytical SQL pipeline against the synthetic dataset 
    using DuckDB and prints summary analytics.
    """
    # 1. Connect to DuckDB (In-Memory)
    con = duckdb.connect(database=':memory:')
    
    # 2. Read the SQL file
    with open('sql/cohort_analytics.sql', 'r') as file:
        query = file.read()
    
    # 3. Execute query and fetch results as DataFrame
    df_results = con.execute(query).fetchdf()
    
    print("\n--- SQL Transformation Pipeline Executed Successfully ---")
    print(f"Processed Rows: {len(df_results)}")
    print("\nTop 5 Rows (Processed Metrics):")
    print(df_results[['week_start', 'total_revenue', 'blended_roas', 'rolling_4wk_avg_revenue', 'effective_paid_spend']].head())
    
    # Save output for reporting or BigQuery comparison
    df_results.to_csv('data/transformed_marketing_metrics.csv', index=False)
    print("\nTransformed metrics saved to 'data/transformed_marketing_metrics.csv'")

if __name__ == "__main__":
    run_sql_pipeline()