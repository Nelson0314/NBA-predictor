import os
import subprocess
import sys
import shutil

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import ROOT_DIR

def run_step(description, command):
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"{'='*50}")
    try:
        subprocess.run(command, check=True, cwd=ROOT_DIR)
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        sys.exit(1)

def main():
    # 1. Update Live Data (2025 Season)
    # This fetches data up to TODAY
    run_step("Updating 2025 Live Data", ["python", "scripts/update_live_2025.py"])
    
    # 2. Fetch Latest Odds
    # Uses src/odds.py to get event_odds_data.json
    run_step("Fetching Latest Odds", ["python", "src/odds.py"])
    
    # 3. Generate/Update Heatmaps
    # Ensures we have heatmaps for the newly fetched games
    run_step("Updating Heatmaps", ["python", "scripts/generate_heatmaps_2025.py"])
    
    # 4. Generate Predictions & Bets
    # Reads event_odds_data.json and generates bets.csv
    run_step("Generating Prediction Report", ["python", "scripts/predict_bets.py"])
    
    print(f"\n{'='*50}")
    print("DONE! Betting report saved to 'bets.csv'.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
