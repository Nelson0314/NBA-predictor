import pandas as pd
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR, ROOT_DIR

# Config
BETS_FILE = os.path.join(ROOT_DIR, 'bets_1226.csv')
ACTUALS_FILE = os.path.join(DATA_DIR, 'live_2025', 'games_2025.csv')
OUTPUT_FILE = os.path.join(ROOT_DIR, 'bets_1226_results.csv')
TARGET_DATE = '2025-12-26'

def main():
    print(f"Evaluating bets from {BETS_FILE} against actuals in {ACTUALS_FILE}...")
    
    if not os.path.exists(BETS_FILE):
        print(f"Error: {BETS_FILE} not found.")
        return
        
    bets_df = pd.read_csv(BETS_FILE)
    actuals_df = pd.read_csv(ACTUALS_FILE)
    
    # Filter bets that are not SKIP
    active_bets = bets_df[bets_df['Pick'] != 'SKIP'].copy()
    print(f"Active Bets to Evaluate: {len(active_bets)}")
    
    # Process Actuals
    # Ensure Date format
    actuals_df['GAME_DATE'] = pd.to_datetime(actuals_df['GAME_DATE'])
    target_dt = pd.to_datetime(TARGET_DATE)
    
    # Filter for target date
    day_games = actuals_df[actuals_df['GAME_DATE'] == target_dt]
    print(f"Found {len(day_games)} player records for {TARGET_DATE}")
    
    if len(day_games) == 0:
        print(f"Warning: No games found for {TARGET_DATE}. Check date format or fetch status.")
        # Try latest date in file
        latest = actuals_df['GAME_DATE'].max()
        print(f"Latest date in file: {latest}")
        return

    # Create Lookup: (Player_Name) -> Row
    # Note: Player names must match. 
    # bets_1226 used names from odds file. games_2025 has 'Player_Name'.
    # We might need fuzzy match or case normalization.
    
    # Normalize
    active_bets['Player_Name_Norm'] = active_bets['Player_Name'].str.lower().str.strip()
    day_games['Player_Name_Norm'] = day_games['Player_Name'].str.lower().str.strip()
    
    results = []
    
    wins = 0
    losses = 0
    pushes = 0
    total_profit = 0.0
    total_wagered = 0.0 # Assuming 1 unit per bet
    
    for idx, row in active_bets.iterrows():
        p_name = row['Player_Name_Norm']
        target = row['Target'] # PTS, AST, REB
        line = float(row['Line'])
        pick = row['Pick'] # OVER, UNDER
        odds = float(row['Pick_Odds'])
        
        # Find actual
        player_rec = day_games[day_games['Player_Name_Norm'] == p_name]
        
        if len(player_rec) == 0:
            # DNP or Name Mismatch
            res_row = row.copy()
            res_row['Actual'] = None
            res_row['Result'] = 'DNP/Void'
            res_row['Profit'] = 0.0
            results.append(res_row)
            continue
            
        # Get Stat
        # Map Target to Column
        # PTS->PTS, AST->AST, REB->REB
        actual_val = player_rec.iloc[0][target]
        
        # Determine Outcome
        outcome = 'LOSS'
        profit = -1.0
        
        if pick == 'OVER':
            if actual_val > line:
                outcome = 'WIN'
                profit = odds - 1.0
            elif actual_val == line:
                outcome = 'PUSH'
                profit = 0.0
        elif pick == 'UNDER':
            if actual_val < line:
                outcome = 'WIN'
                profit = odds - 1.0
            elif actual_val == line:
                outcome = 'PUSH'
                profit = 0.0
                
        # Update Stats
        if outcome == 'WIN': wins += 1
        elif outcome == 'LOSS': losses += 1
        elif outcome == 'PUSH': pushes += 1
        
        if outcome != 'DNP/Void':
            total_profit += profit
            total_wagered += 1.0
            
        res_row = row.copy()
        res_row['Actual'] = actual_val
        res_row['Result'] = outcome
        res_row['Profit'] = round(profit, 2)
        results.append(res_row)
        
    # Summary
    final_df = pd.DataFrame(results)
    
    print("\n" + "="*30)
    print("EVALUATION REPORT")
    print("="*30)
    print(f"Total Reviewed: {len(active_bets)}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Pushes: {pushes}")
    print(f"Void/DNP: {len(active_bets) - wins - losses - pushes}")
    
    if total_wagered > 0:
        roi = (total_profit / total_wagered) * 100
        win_rate = (wins / (wins + losses)) * 100 if (wins+losses) > 0 else 0
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Profit (Units): {total_profit:.2f}")
        print(f"ROI: {roi:.1f}%")
    else:
        print("No settled bets.")
        
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDetailed results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
