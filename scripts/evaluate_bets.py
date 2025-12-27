import pandas as pd
import numpy as np
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR, ROOT_DIR

BETS_FILE = os.path.join(ROOT_DIR, 'bets.csv')
GAMES_FILE = os.path.join(DATA_DIR, 'live_2025', 'games_2025.csv')
OUTPUT_FILE = os.path.join(ROOT_DIR, 'bets_result.csv')

def main():
    if not os.path.exists(BETS_FILE):
        print(f"Error: {BETS_FILE} not found.")
        return

    if not os.path.exists(GAMES_FILE):
        print(f"Error: {GAMES_FILE} not found (need actuals to verify).")
        return

    print("Loading bets and actuals...")
    bets_df = pd.read_csv(BETS_FILE)
    games_df = pd.read_csv(GAMES_FILE)
    
    # Ensure Dates match format
    # bets.csv 'Date' is YYYY-MM-DD
    # games.csv 'GAME_DATE' is YYYY-MM-DD (from update_live_2025)
    
    if 'Date' not in bets_df.columns:
        print("Error: 'Date' column missing in bets.csv. Please regenerate bets using updated predict_bets.py.")
        return

    bets_df['Date'] = pd.to_datetime(bets_df['Date'])
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    
    # Filter for active bets
    if 'Pick' in bets_df.columns:
        active_bets = bets_df[bets_df['Pick'] != 'SKIP'].copy()
    else:
        active_bets = bets_df.copy() # Assume all are bets if no Pick col? No, unsafe.
    
    if active_bets.empty:
        print("No active bets found to verify.")
        return

    print(f"Verifying {len(active_bets)} bets...")
    
    results = []
    
    # Pre-index games for speed
    # unique key: (Player_Name, GAME_DATE)
    # Note: Player_Name in games_df might differ slightly? 
    # 'Player_Name' in games_2025 comes from NBA API.
    # 'Player_Name' in bets comes from FanDuel via odds.py.
    # We might need fuzzy match. For now try exact.
    
    games_map = {}
    for idx, row in games_df.iterrows():
        k = (row['Player_Name'], row['GAME_DATE'])
        games_map[k] = row
        
    verified_count = 0
    
    for idx, bet in active_bets.iterrows():
        p_name = bet['Player_Name']
        date = bet['Date']
        target = bet['Target']
        line = bet['Line']
        pick = bet['Pick'] # OVER / UNDER
        odds = bet['Pick_Odds']
        
        # Fair Odds Logic (for No-Vig ROI)
        # Use FairOdds_Over/Under from csv
        fair_odds = 0.0
        if pick == 'OVER':
            fair_odds = bet.get('FairOdds_Over', 2.0)
        elif pick == 'UNDER':
            fair_odds = bet.get('FairOdds_Under', 2.0)
            
        # Find Actual Game
        key = (p_name, date)
        
        # Attempt match
        match = None
        if key in games_map:
            match = games_map[key]
        else:
            # Try fuzzy matching? OR just skip
            # Often names differ: "Luka Doncic" vs "Luka Dončić"
            # simple check
            found = False
            for (gn, gd), grow in games_map.items():
                if gd == date and (p_name in gn or gn in p_name):
                    match = grow
                    found = True
                    break
            if not found:
                bet['Result'] = 'PENDING/MISSING'
                results.append(bet)
                continue
                
        verified_count += 1
        
        # Calculate Actual Value
        # Targets: PTS, REB, AST, PTS+REB+AST, etc.
        actual_val = 0
        
        # Composite Parser
        # Split by '+'
        components = target.split('+')
        valid_comp = True
        for c in components:
            if c not in match:
                valid_comp = False
                break
            actual_val += match[c]
            
        if not valid_comp:
            bet['Result'] = 'ERROR_STAT_MISSING'
            results.append(bet)
            continue
            
        bet['Actual'] = actual_val
        
        # Determine Win/Loss
        outcome = 'PUSH'
        profit_real = 0.0
        profit_no_vig = 0.0
        
        if pick == 'OVER':
            if actual_val > line:
                outcome = 'WIN'
                profit_real = odds - 1.0
                profit_no_vig = fair_odds - 1.0
            elif actual_val < line:
                outcome = 'LOSS'
                profit_real = -1.0
                profit_no_vig = -1.0
        elif pick == 'UNDER':
            if actual_val < line:
                outcome = 'WIN'
                profit_real = odds - 1.0
                profit_no_vig = fair_odds - 1.0
            elif actual_val > line:
                outcome = 'LOSS'
                profit_real = -1.0
                profit_no_vig = -1.0
                
        bet['Result'] = outcome
        bet['Profit_Real'] = profit_real
        bet['Profit_NoVig'] = profit_no_vig
        
        results.append(bet)
        
    # Summary
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_FILE, index=False)
    
    completed = res_df[res_df['Result'].isin(['WIN', 'LOSS', 'PUSH'])]
    
    if completed.empty:
        print("No completed bets found (maybe games haven't happened?).")
        return
        
    wins = len(completed[completed['Result'] == 'WIN'])
    total = len(completed)
    win_rate = (wins / total) * 100
    
    roi_real = (completed['Profit_Real'].sum() / total) * 100
    roi_novig = (completed['Profit_NoVig'].sum() / total) * 100
    
    print(f"\n{'='*40}")
    print(f"BETTING EVALUATION REPORT")
    print(f"{'='*40}")
    print(f"Verified Bets: {total}/{len(active_bets)}")
    print(f"Win Rate:      {wins}/{total} ({win_rate:.1f}%)")
    print(f"ROI (Real):    {roi_real:.1f}% (with Vig)")
    print(f"ROI (No Vig):  {roi_novig:.1f}% (Fair Odds)")
    print(f"{'='*40}")
    print(f"Detailed results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
