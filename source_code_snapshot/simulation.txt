
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
    CONF_THRESH = 40.0
    MAX_SPREADS = {'PTS': 30.0, 'AST': 8.0, 'REB': 10.0}
    
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
    
    # Volatility & Bias Analysis
    report.append("\n--- Volatility & Bias Analysis ---")
    
    # Calculate Avg Spread & Implied Std for first target (PTS usually)
    # y_true is (N, Targets). gambler_preds is (N, Targets, 3)
    avg_spread = np.mean(gambler_preds[:, 0, 2] - gambler_preds[:, 0, 0])
    avg_std_implied = avg_spread / 2.56
    actual_std = np.std(y_true[:, 0])
    
    report.append(f"Avg Spread (P90-P10) [PTS]: {avg_spread:.2f}")
    report.append(f"Avg Implied Std [PTS]: {avg_std_implied:.2f}")
    report.append(f"Actual Std [PTS]: {actual_std:.2f}")
    report.append(f"Capture Ratio: {avg_std_implied/actual_std:.2f}")
    report.append("-" * 40)
    
    # Per-Target Analysis
    from sklearn.metrics import mean_absolute_error
    
    for t_idx, t_name in enumerate(target_cols):
        # Gambler Stats
        g_mae = mean_absolute_error(y_true[:, t_idx], gambler_preds[:, t_idx, 1])
        g_bias = np.mean(y_true[:, t_idx] - gambler_preds[:, t_idx, 1])
        
        # House Stats
        h_mae = mean_absolute_error(y_true[:, t_idx], house_preds[:, t_idx])
        h_bias = np.mean(y_true[:, t_idx] - house_preds[:, t_idx])
        
        g_bias_str = "Underest" if g_bias > 0 else "Overest"
        h_bias_str = "Underest" if h_bias > 0 else "Overest"
        
        report.append(f"{t_name} | Our MAE: {g_mae:.2f} (Bias: {g_bias:.2f} {g_bias_str}) | House MAE: {h_mae:.2f} (Bias: {h_bias:.2f} {h_bias_str})")

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

            # ---------------------------------------------------------
            # 2. Confidence Analysis Check
            # ---------------------------------------------------------
            # Bin by 'GamblerSpread' (smaller spread = higher confidence)
            # Or by 'ConfPercent' which we calculated.
            
            # Using GamblerSpread as raw confidence metric
            placed_df['SpreadBin'] = pd.qcut(placed_df['GamblerSpread'], q=5, duplicates='drop')
            
            # Group Stats
            bin_stats = placed_df.groupby('SpreadBin', observed=True).agg({
                'Outcome': lambda x: (x > 0).mean(), # Win Rate
                'Bankroll': 'count' # Bet Count
            }).rename(columns={'Outcome': 'WinRate', 'Bankroll': 'Count'})
            
            # Calculate Avg PnL per bin
            bin_stats['AvgPnL'] = placed_df.groupby('SpreadBin', observed=True)['Outcome'].mean()

            # Plot Confidence Analysis
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Spread Bin (Lower Spread = Higher Confidence)')
            ax1.set_ylabel('Win Rate', color=color)
            bin_stats['WinRate'].plot(kind='bar', ax=ax1, color=color, alpha=0.6, position=0, width=0.4)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.axhline(0.524, color='red', linestyle='--', label='Breakeven (52.4%)')
            
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('Avg PnL ($)', color=color)
            bin_stats['AvgPnL'].plot(kind='line', ax=ax2, color=color, marker='o', linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Win Rate & Profitability by Confidence')
            fig.tight_layout()
            plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'))
            plt.close()
            
    # ---------------------------------------------------------
    # 3. Accuracy Scatter Plot
    # ---------------------------------------------------------
    # Plot Actual vs Gambler vs House for a sample
    sample_df = log_df.sample(min(10000, len(log_df))) if len(log_df) > 0 else log_df
    if not sample_df.empty:
        targets = sample_df['Target'].unique()
        fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 6))
        if len(targets) == 1: axes = [axes] # Handle single target case
        
        for idx, t in enumerate(targets):
            subset = sample_df[sample_df['Target'] == t]
            ax = axes[idx]
            
            # Plot
            ax.scatter(subset['Actual'], subset['Line'], alpha=0.3, label='House Line', color='red', s=10)
            ax.scatter(subset['Actual'], subset['GamblerP50'], alpha=0.3, label='Gambler Pred', color='blue', s=10)
            
            # Perfect Line
            max_val = max(subset['Actual'].max(), subset['GamblerP50'].max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            ax.set_title(f"{t} Predictions")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accuracy_scatter.png'))
        plt.close()
    
    return bankroll, total_bets, win_rate
