import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def run_script(script_relative_path):
    """Executes a Python script in the current virtual environment."""
    script_path = BASE_DIR / script_relative_path
    if not script_path.exists():
        print(f"❌ Error: Script not found at {script_path}")
        sys.exit(1)
        
    print(f"\n🚀 Executing: {script_relative_path}...")
    result = subprocess.run([sys.executable, str(script_path)])
    
    if result.returncode != 0:
        print(f"❌ Execution failed for {script_relative_path}")
        sys.exit(result.returncode)
    print(f"✅ Completed: {script_relative_path}")

def launch_dashboard():
    """Launches the Streamlit executive dashboard."""
    dashboard_path = BASE_DIR / "scripts" / "11_streamlit_dashboard.py"
    print(f"\n📊 Launching Streamlit Dashboard from {dashboard_path.name}...")
    subprocess.run(["streamlit", "run", str(dashboard_path)])

def main():
    parser = argparse.ArgumentParser(
        description="Marketing Measurement Consultancy Stack - End-to-End CLI Pipeline"
    )
    
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run Phase 11 Synthetic Control & Causally Calibrated PyMC MMM"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Export diagnostic visual figures to reports/figures/"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the Streamlit Scenario Planner Dashboard"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute full pipeline: Calibrate -> Export Figures -> Launch Dashboard"
    )

    args = parser.parse_args()

    if not any([args.calibrate, args.report, args.serve, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.calibrate or args.all:
        run_script("scripts/12_causal_synthetic_control_calibration.py")

    if args.report or args.all:
        run_script("scripts/export_report_figures.py")

    if args.serve or args.all:
        launch_dashboard()

if __name__ == "__main__":
    main()