
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR, ROOT_DIR

GAMES_FILE = os.path.join(DATA_DIR, 'live_2025', 'games_2025.csv')

MODELS_TO_EVAL = [
    ('MultiModal', 'bets.csv'),
    ('Naive', 'naive_bet.csv'),
    ('LinearReg', 'linearRegression_bet.csv'),
    ('XGBoost', 'xgBoost_bet.csv'),
    ('Hybrid', 'hybrid_bet.csv')
]

def normalize_date(d):
    """Encodes date to YYYY-MM-DD string safely."""
    try:
        if pd.isna(d): return None
        return pd.to_datetime(d).strftime('%Y-%m-%d')
    except:
        return str(d).split(' ')[0]

def evaluate_dataframe(df, games_map, method_name):
    results = []
    
    if 'Pick' in df.columns:
        active_bets = df[df['Pick'] != 'SKIP'].copy()
    else:
        active_bets = df.copy()

    if active_bets.empty: return pd.DataFrame()

    for idx, bet in active_bets.iterrows():
        p_name = bet.get('Player_Name')
        date_raw = bet.get('Date')
        bet_date = normalize_date(date_raw)
        
        target = bet.get('Target')
        line = bet.get('Line')
        pick = bet.get('Pick')
        odds = bet.get('Pick_Odds', 0.0)
        
        fair_odds = 0.0
        if pick == 'OVER': fair_odds = bet.get('FairOdds_Over', 2.0)
        elif pick == 'UNDER': fair_odds = bet.get('FairOdds_Under', 2.0)
        if pd.isna(fair_odds) or fair_odds <= 0: fair_odds = 2.0

        # Match Logic
        match = None
        key = (p_name, bet_date)
        match = games_map.get(key)
        
        # Fuzzy
        if match is None:
            # Maybe Name mismatch?
            # Try iterating specific date entries to save time?
            # Or iterate all (slow)
            for (gn, gd), grow in games_map.items():
                if gd == bet_date and (p_name in gn or gn in p_name):
                    match = grow
                    break
        
        if match is None:
            bet['Result'] = 'PENDING/MISSING'
            bet['Method'] = method_name
            results.append(bet)
            continue
            
        # Calculate Actual
        actual_val = 0
        components = target.split('+')
        valid_comp = True
        for c in components:
            if c not in match:
                valid_comp = False
                break
            actual_val += match[c]
            
        if not valid_comp:
            bet['Result'] = 'ERROR_STAT_MISSING'
            bet['Method'] = method_name
            results.append(bet)
            continue
            
        bet['Actual'] = actual_val
        
        # Determine Outcome
        outcome = 'PUSH'
        profit_real = 0.0
        profit_no_vig = 0.0
        
        if pick == 'OVER':
            if actual_val > line:
                outcome = 'WIN'; profit_real = odds - 1.0; profit_no_vig = fair_odds - 1.0
            elif actual_val < line:
                outcome = 'LOSS'; profit_real = -1.0; profit_no_vig = -1.0
        elif pick == 'UNDER':
            if actual_val < line:
                outcome = 'WIN'; profit_real = odds - 1.0; profit_no_vig = fair_odds - 1.0
            elif actual_val > line:
                outcome = 'LOSS'; profit_real = -1.0; profit_no_vig = -1.0
                
        bet['Result'] = outcome
        bet['Profit_Real'] = profit_real
        bet['Profit_NoVig'] = profit_no_vig
        bet['Method'] = method_name
        
        results.append(bet)
        
    return pd.DataFrame(results)

def main():
    if not os.path.exists(GAMES_FILE):
        print(f"Error: {GAMES_FILE} not found.")
        return

    print("Loading Ground Truth (Games 2025)...")
    games_df = pd.read_csv(GAMES_FILE)
    if 'GAME_DATE' in games_df.columns:
        # Standardize date
        games_df['GAME_DATE_STR'] = pd.to_datetime(games_df['GAME_DATE']).dt.strftime('%Y-%m-%d')
    else:
        print("Error: GAME_DATE missing.")
        return

    # Build Map (Player, DateStr) -> Row
    games_map = {}
    for idx, row in games_df.iterrows():
        k = (row.get('Player_Name'), row.get('GAME_DATE_STR'))
        games_map[k] = row
        
    all_results = []
    
    # Determine Report Date from bets.csv (MultiModal) or baseline
    bet_date_suffix = datetime.now().strftime('%Y-%m-%d') # Fallback
    
    # Pre-scan to find primary date
    for _, fname in MODELS_TO_EVAL:
        fpath = os.path.join(ROOT_DIR, fname)
        if os.path.exists(fpath):
            try:
                tmp = pd.read_csv(fpath)
                if 'Date' in tmp.columns and not tmp.empty:
                    # Get most frequent date
                    dates = pd.to_datetime(tmp['Date']).dt.strftime('%Y-%m-%d')
                    top_date = dates.mode()[0]
                    bet_date_suffix = top_date
                    print(f"Detected Report Date from {fname}: {bet_date_suffix}")
                    break
            except: pass

    output_csv = os.path.join(ROOT_DIR, f'bets_result_{bet_date_suffix}.csv')
    output_txt = os.path.join(ROOT_DIR, f'evaluation_report_{bet_date_suffix}.txt')
    
    summary_lines = []
    header = f"{'Method':<15} | {'Bets':<6} | {'Win Rate':<10} | {'ROI (Real)':<12} | {'ROI (NoVig)':<12}"
    
    summary_lines.append(f"BETTING EVALUATION REPORT (Date: {bet_date_suffix})")
    summary_lines.append("="*65)
    summary_lines.append(header)
    summary_lines.append("-" * 65)
    print("\n" + "\n".join(summary_lines[-3:]))

    for method, fname in MODELS_TO_EVAL:
        fpath = os.path.join(ROOT_DIR, fname)
        if not os.path.exists(fpath):
            line = f"{method:<15} | [FILE NOT FOUND]"
            print(line); summary_lines.append(line)
            continue
            
        df = pd.read_csv(fpath)
        eval_df = evaluate_dataframe(df, games_map, method)
        
        if eval_df.empty:
            line = f"{method:<15} | [NO BETS]"
            print(line); summary_lines.append(line)
            continue
            
        all_results.append(eval_df)
        
        completed = eval_df[eval_df['Result'].isin(['WIN', 'LOSS', 'PUSH'])]
        
        # Debug why empty
        # if completed.empty:
        #     print(f"DEBUG: {method} - Input Bets: {len(eval_df)}, Results: {eval_df['Result'].unique()}")
        #     if not eval_df.empty:
        #         sample = eval_df.iloc[0]
        #         print(f"  Sample Mismatch: Bet({sample['Player_Name']}, {sample.get('Date')}) vs Map Keys? ")
        
        if completed.empty:
            line = f"{method:<15} | [NO COMPLETED BETS]"
            print(line); summary_lines.append(line)
            continue
            
        wins = len(completed[completed['Result'] == 'WIN'])
        total = len(completed)
        win_rate = (wins / total) * 100
        roi_real = (completed['Profit_Real'].sum() / total) * 100
        roi_novig = (completed['Profit_NoVig'].sum() / total) * 100
        
        line = f"{method:<15} | {total:<6} | {win_rate:.1f}%     | {roi_real:+.1f}%      | {roi_novig:+.1f}%"
        print(line); summary_lines.append(line)
        
    summary_lines.append("="*65)
    print("="*65)
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"\nDetailed Results Saved to: {output_csv}")
    
    with open(output_txt, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Summary Report Saved to: {output_txt}")

if __name__ == "__main__":
    main()
