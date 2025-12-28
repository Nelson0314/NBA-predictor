
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import json

from src.odds import calculate_odds

def run_betting_simulation(gambler_preds, house_preds, y_true, metadata, target_cols, save_dir, plot_confidence=True):
    """
    Run betting simulation comparing Gambler (P10, P50, P90) vs House (Line).
    
    Args:
        gambler_preds: (N, n_targets, 3) - [P10, P50, P90], Inverse Transformed (Original Scale).
        house_preds: (N, n_targets) - The Betting Line, Inverse Transformed (Original Scale).
        y_true: (N, n_targets) - Actual outcomes.
        metadata: (N,) - List/Array of PlayerIDs or identifying info.
        target_cols: List of target names ['PTS', 'AST', '...'].
        save_dir: Directory to save reports and plots.
    """
    
    print("Simulating Betting...")
    bankroll = 10000
    bet_history = []
    total_bets = 0
    wins = 0
    losses = 0
    
    # Constants
    CONF_THRESH = 60.0
    MAX_SPREADS = {'PTS': 20.0, 'AST': 8.0, 'REB': 10.0}
    
    # Iterate through all samples
    num_samples = len(y_true)
    for i in range(num_samples):
        for t_idx, t_name in enumerate(target_cols):
            # House Line
            line = house_preds[i, t_idx]
            
            # Gambler Preds
            g_p10 = gambler_preds[i, t_idx, 0]
            g_p50 = gambler_preds[i, t_idx, 1]
            g_p90 = gambler_preds[i, t_idx, 2]
            
            # 1. Calculate Confidence
            g_spread = g_p90 - g_p10
            max_spread = MAX_SPREADS.get(t_name, 20.0)
            conf_percent = max(0.0, 100.0 * (1.0 - (g_spread / max_spread)))
            
            # 2. Calculate Odds (House vs Line)
            # Standard Deviation assumptions for Odds Maker
            scale_std = 9.0
            if t_name == 'AST': scale_std = 3.0
            if t_name == 'REB': scale_std = 4.0
            
            # Note: house_pred IS the line in this context
            odds_over, odds_under, _ = calculate_odds(line, line, std_dev=scale_std) 
            # Ideally calculate_odds uses (HousePred, Line). If HousePred == Line, odds are ~1.91 (50/50).
            
            # 3. Gambler EV
            # Est. Gambler Std Dev
            g_std = g_spread / 2.56 if g_spread > 0 else 1.0
            
            # Z-Score of Line regarding Gambler Distribution
            g_z = (line - g_p50) / g_std
            g_prob_over = 1 - norm.cdf(g_z)
            g_prob_under = 1.0 - g_prob_over
            
            ev_over = (g_prob_over * odds_over) - 1
            ev_under = (g_prob_under * odds_under) - 1
            
            # 4. Place Bet
            bet_size = 100
            bet_placed = False
            bet_type = "NONE"
            bet_odds = 0.0
            status = "SKIPPED"
            reason = ""
            
            if conf_percent >= CONF_THRESH:
                if ev_over > 0.05:
                    bet_type = "OVER"
                    bet_odds = odds_over
                    bet_placed = True
                    status = "PLACED"
                elif ev_under > 0.05:
                    bet_type = "UNDER"
                    bet_odds = odds_under
                    bet_placed = True
                    status = "PLACED"
                else:
                    reason = "EV_LOW"
                    status = "SKIPPED_EV"
            else:
                reason = f"CONF_LOW ({conf_percent:.1f}%)"
                status = "SKIPPED_CONF"
                
            # 5. Outcome
            outcome = 0
            actual = y_true[i, t_idx]
            
            if bet_placed:
                if bet_type == "OVER":
                    if actual > line:
                        outcome = bet_size * (bet_odds - 1)
                        wins += 1
                    else:
                        outcome = -bet_size
                        losses += 1
                else: # UNDER
                    if actual < line:
                        outcome = bet_size * (bet_odds - 1)
                        wins += 1
                    else:
                        outcome = -bet_size
                        losses += 1
                
                bankroll += outcome
                total_bets += 1
                
            # Log
            pid = metadata[i] if i < len(metadata) else "Unknown"
            bet_history.append({
                'PlayerID': pid,
                'Target': t_name,
                'Line': line,
                'Actual': actual,
                'GamblerP50': g_p50,
                'GamblerSpread': g_spread,
                'ConfPercent': conf_percent,
                'ProbOver': g_prob_over,
                'OddsOver': odds_over,
                'EV_Over': ev_over,
                'BetType': bet_type,
                'Outcome': outcome,
                'Bankroll': bankroll,
                'Status': status,
                'Reason': reason
            })

    # Save Log
    log_df = pd.DataFrame(bet_history)
    log_path = os.path.join(save_dir, 'betting_log.csv')
    log_df.to_csv(log_path, index=False)
    print(f"Betting Log Saved: {log_path}")
    
    # ---------------------------------------------------------
    # Reports & Plots
    # ---------------------------------------------------------
    roi = ((bankroll - 10000) / (total_bets * 100)) * 100 if total_bets > 0 else 0
    win_rate = (wins / total_bets) * 100 if total_bets > 0 else 0
    
    report = []
    report.append("========================================")
    report.append("SIMULATION REPORT")
    report.append("========================================")
    report.append(f"Total Bets: {total_bets}")
    report.append(f"Wins: {wins} | Losses: {losses}")
    report.append(f"Win Rate: {win_rate:.2f}%")
    report.append(f"Final Bankroll: ${bankroll:.2f} (Start $10000)")
    report.append(f"ROI: {roi:.2f}%")
    report.append("-" * 40)
    
    report_path = os.path.join(save_dir, 'simulation_report.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    print(f"Report Saved: {report_path}")

    # Plot PnL
    plt.figure(figsize=(10, 6))
    if total_bets > 0:
        # Filter for placed bets only for cumulative PnL
        placed_df = log_df[log_df['Status'] == 'PLACED'].copy()
        if not placed_df.empty:
            placed_df['Outcome'] = pd.to_numeric(placed_df['Outcome'])
            placed_df['CumulativePnL'] = placed_df['Outcome'].cumsum()
            plt.plot(range(len(placed_df)), placed_df['CumulativePnL'], label='Gambler PnL')
            plt.axhline(0, color='r', linestyle='--', label='Break Even')
            plt.title('Bankroll Simulation (Gambler vs House)')
            plt.xlabel('Bet Number')
            plt.ylabel('PnL ($)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'pnl_chart.png'))
            plt.close()
    
    return bankroll, total_bets, win_rate
