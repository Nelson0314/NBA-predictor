
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from scipy.stats import norm
from tqdm import tqdm
import warnings
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DATA_DIR, SAVED_MODELS_DIR, ROOT_DIR
from src.seqModel import NbaTransformer, loadAndPreprocessData, createSequences
from src.odds import fetch_odds

warnings.filterwarnings('ignore')

# Config
DATASET_DIR_OLD = DATA_DIR
DATASET_DIR_NEW = os.path.join(DATA_DIR, "live_2025")
ODDS_FILE = os.path.join(os.path.dirname(__file__), "..", "event_odds_data.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "bets.csv")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 0. Model Discovery & Config
# ==========================================
def find_best_model(base_dir):
    seq_dir = os.path.join(base_dir, "seq")
    if not os.path.exists(seq_dir): seq_dir = base_dir
        
    print(f"Searching for best model in {seq_dir}...")
    best_path = None
    best_score = float('inf')
    
    # Check if config exists directly (case where files are flat in seq dir)
    if os.path.exists(os.path.join(seq_dir, 'config.json')):
        print(f"Found model directly in: {seq_dir}")
        return seq_dir

    candidates = [os.path.join(seq_dir, d) for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))]
    
    if not candidates:
        print("No model folders found.")
        return None
        
    for p in candidates:
        c_path = os.path.join(p, 'config.json')
        if os.path.exists(c_path):
            try:
                with open(c_path, 'r') as f:
                    cfg = json.load(f)
                    score = cfg.get('valid_loss', float('inf'))
                    if score < best_score:
                        best_score = score
                        best_path = p
            except: pass
    
    if best_path:
        print(f"Selected Best Model: {os.path.basename(best_path)} (Loss: {best_score:.4f})")
    else:
        best_path = max(candidates, key=os.path.getmtime)
        print(f"Selected Recent Model: {os.path.basename(best_path)}")
        
    return best_path

MODEL_PATH = find_best_model(SAVED_MODELS_DIR)
if not MODEL_PATH:
    print("CRITICAL: No models found. Exiting.")
    exit()

with open(os.path.join(MODEL_PATH, 'config.json'), 'r') as f:
    CONFIG = json.load(f)

SEQ_LENGTH = CONFIG.get('seqLength', 7)
print(f"Using SEQ_LENGTH: {SEQ_LENGTH}")

# ==========================================
# 1. Parsing Odds
# ==========================================
def parse_odds(file_path):
    print(f"Parsing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    for game in data:
        commence = game.get('commence_time')
        bookie = next((b for b in game['bookmakers'] if b['key'] == 'fanduel'), None)
        if not bookie: bookie = game['bookmakers'][0] if game['bookmakers'] else None
        if not bookie: continue
            
        for market in bookie['markets']:
            m_key = market['key']
            target_map = {
                'player_points': 'PTS', 'player_assists': 'AST', 'player_rebounds': 'REB',
                'player_points_rebounds_assists': 'PTS+REB+AST',
                'player_points_rebounds': 'PTS+REB', 'player_points_assists': 'PTS+AST',
                'player_rebounds_assists': 'REB+AST'
            }
            if m_key not in target_map: continue
            tgt = target_map[m_key]
            
            lines_dict = {}
            for out in market['outcomes']:
                k = (out['description'], out['point'])
                if k not in lines_dict: lines_dict[k] = {}
                lines_dict[k][out['name']] = out['price']
            
            for (p_name, point), prices in lines_dict.items():
                if 'Over' in prices and 'Under' in prices:
                    o_price = prices['Over']
                    u_price = prices['Under']
                    p_over = 1/o_price
                    p_under = 1/u_price
                    margin = p_over + p_under
                    fair_p_over = p_over / margin
                    fair_p_under = p_under / margin
                    
                    try:
                        c_iso = commence.replace('Z', '+00:00')
                        dt_utc = datetime.fromisoformat(c_iso)
                        dt_et = dt_utc.astimezone(timezone(timedelta(hours=-5)))
                        game_date = dt_et.strftime('%Y-%m-%d')
                    except:
                        game_date = commence.split('T')[0] if commence else "Unknown"

                    rows.append({
                        'Date': game_date, 'Player_Name': p_name, 'Target': tgt, 'Line': point,
                        'Odds_Over': o_price, 'Odds_Under': u_price,
                        'FairPro_Over': round(fair_p_over, 3), 'FairPro_Under': round(fair_p_under, 3),
                        'FairOdds_Over': round(1/fair_p_over, 2), 'FairOdds_Under': round(1/fair_p_under, 2)
                    })
    return pd.DataFrame(rows)

# ==========================================
# 2. Baseline Helper
# ==========================================
def train_baselines(games_all, feat_cols, target_cols):
    print("Training Baseline Models (Naive, LR, XGB)...")
    
    # 1. Create Sequences
    # Use createSequences from seqModel (returns x, y, meta)
    # x: (N, Seq, F)
    x, y, _ = createSequences(games_all, SEQ_LENGTH, feat_cols, target_cols)
    
    # Flatten x for LR/XGB
    N, S, F = x.shape
    x_flat = x.reshape(N, S*F)
    
    # Y is usually (N, 3) for PTS, AST, REB
    # Make sure we only take first 3 if more exist
    y_target = y[:, :3]
    
    # Train LR
    lr = LinearRegression()
    lr.fit(x_flat, y_target)
    
    # Train XGB
    xgb_models = []
    for i in range(3):
        m = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1)
        m.fit(x_flat, y_target[:, i])
        xgb_models.append(m)
        
    # Calculate Residuals (Stds) for Direct Comparison logic
    lr_pred = lr.predict(x_flat)
    lr_stds = [np.std(y_target[:, i] - lr_pred[:, i]) for i in range(3)]
    
    xgb_pred = np.column_stack([m.predict(x_flat) for m in xgb_models])
    xgb_stds = [np.std(y_target[:, i] - xgb_pred[:, i]) for i in range(3)]
    
    return lr, lr_stds, xgb_models, xgb_stds

# ==========================================
# 3. Main Logic
# ==========================================
def main():
    # Only fetch if missing to avoid redundancy in autobet
    if not os.path.exists(ODDS_FILE):
        print("Odds file missing. Fetching...")
        try: fetch_odds()
        except Exception as e: print(f"Odds Fetch Warning: {e}")
    else:
        print("Using existing odds data (Skipping fetch)...")
        
    odds_df = parse_odds(ODDS_FILE)
    if odds_df.empty: print("No odds found."); return

    # Load All Data
    print("Loading and Merging Data...")
    gamesPathOld = os.path.join(DATASET_DIR_OLD, 'games.csv')
    gamesPathNew = os.path.join(DATASET_DIR_NEW, 'games_2025.csv')
    
    # Concatenate raw CSVs then load to handle rolling features properly
    temp_path = os.path.join(DATA_DIR, "combined_temp.csv")
    df1 = pd.read_csv(gamesPathOld, low_memory=False)
    if os.path.exists(gamesPathNew):
        df2 = pd.read_csv(gamesPathNew, low_memory=False)
        df_all = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=['GAME_ID', 'Player_ID'])
    else:
        df_all = df1
        
    df_all.to_csv(temp_path, index=False)
    
    # Load and Preprocess (Calculates Rolling Stats)
    gamesAll, featureCols, targetCols = loadAndPreprocessData(temp_path, SEQ_LENGTH)
    try: os.remove(temp_path)
    except: pass

    # --- Train Baselines ---
    lr_model, lr_stds, xgb_models, xgb_stds = train_baselines(gamesAll, featureCols, targetCols)
    
    # --- Load Sequence Model ---
    print("Loading Sequence Model...")
    scaler = StandardScaler()
    scaler.fit(gamesAll[featureCols].values)
    
    scalerY = MinMaxScaler()
    scalerY.fit(gamesAll[['PTS', 'AST', 'REB']].values)
    
    statEmbedDim = CONFIG.get('statEmbedDim', 128)
    model = NbaTransformer(
        inputDim=len(featureCols), dModel=CONFIG['dModel'], nHead=CONFIG['nHead'], 
        numLayers=CONFIG['numLayers'], outputDim=3, statEmbedDim=statEmbedDim, dropout=CONFIG['dropout']
    ).to(DEVICE)
    
    try: model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model.ckpt'), map_location=DEVICE))
    except Exception as e: print(f"Error loading seq model: {e}"); return
    model.eval()
    
    # --- Prediction Loop ---
    print("Generating Predictions (All Models)...")
    
    seq_bets = []
    naive_bets = []
    lr_bets = []
    xgb_bets = []
    hybrid_bets = []
    
    idx_pts = featureCols.index('PTS')
    idx_ast = featureCols.index('AST')
    idx_reb = featureCols.index('REB')
    
    player_groups = gamesAll.sort_values('GAME_DATE').groupby('Player_Name')
    
    for idx, row in tqdm(odds_df.iterrows(), total=len(odds_df)):
        p_name = row['Player_Name']
        if p_name not in player_groups.groups: continue
            
        p_data = player_groups.get_group(p_name)
        if len(p_data) < SEQ_LENGTH: continue
        
        # Input Data
        seq_data = p_data.iloc[-SEQ_LENGTH:].copy()
        feats = seq_data[featureCols].values # (Seq, F)
        
        # 1. Sequence Model Predict
        feats_scaled = scaler.transform(feats)
        feats_tensor = torch.tensor(feats_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            preds = model(feats_tensor) # (1, 3, 3)
            
        preds = preds.cpu().numpy().reshape(1, 3, 3)
        pred_vals_seq = np.zeros((3, 3))
        for q in range(3):
            inv = scalerY.inverse_transform(preds[:, :, q])
            pred_vals_seq[:, q] = inv[0]
            
        # 2. Baseline Predict
        feats_flat = feats.reshape(1, -1)
        
        # Naive: Mean of last 7 games
        seq_means = np.mean(feats, axis=0) # (F,)
        p_naive = seq_means[[idx_pts, idx_ast, idx_reb]]
        s_naive = np.std(feats, axis=0)[[idx_pts, idx_ast, idx_reb]] # Std of sequence
        
        # LR
        p_lr = lr_model.predict(feats_flat)[0]
        
        # XGB
        p_xgb = np.array([m.predict(feats_flat)[0] for m in xgb_models])
        
        # --- Process Logic Helpers ---
        
        # A. Sequence Logic (Gaussian EV)
        def process_seq(vals, row_data):
            # vals: (3, 3) -> [Target, Quantile]
            row = row_data.to_dict()
            nm = row['Target']
            
            def get_stats(tgt):
                if tgt not in ['PTS','AST','REB']: return 0, 1
                t_idx = ['PTS','AST','REB'].index(tgt)
                p10, p50, p90 = vals[t_idx]
                mean = p50
                std = (p90 - p10) / 2.56
                if std < 0.1: std = 0.1
                return mean, std
                
            if '+' in nm:
                pm, pv = 0.0, 0.0
                for c in nm.split('+'):
                    m, s = get_stats(c)
                    pm += m; pv += s**2
                ps = np.sqrt(pv)
            else:
                pm, ps = get_stats(nm)
                
            z = (row['Line'] - pm) / ps
            p_under = norm.cdf(z)
            p_over = 1.0 - p_under
            
            ev_over = (p_over * row['Odds_Over']) - 1.0
            ev_under = (p_under * row['Odds_Under']) - 1.0
            
            choice = "SKIP"; conf = 0.0; ev = 0.0; odds = 0.0
            if ev_over > 0.05:
                choice = "OVER"; conf = p_over; ev = ev_over; odds = row['Odds_Over']
            elif ev_under > 0.05:
                choice = "UNDER"; conf = p_under; ev = ev_under; odds = row['Odds_Under']
                
            row.update({'Pred_Mean': round(pm,1), 'Pred_Std': round(ps,2), 'Pick': choice, 'Pick_EV': round(ev,3), 'Pick_Odds': odds})
            return row

        # B. Baseline Logic (Direct Compare)
        def process_base(p_vals, s_vals, row_data):
            row = row_data.to_dict()
            val_map = {'PTS':0, 'AST':1, 'REB':2}
            mu = 0.0
            if '+' in row['Target']:
                for c in row['Target'].split('+'): mu += p_vals[val_map[c]]
            else:
                mu = p_vals[val_map[row['Target']]]
                
            diff = mu - row['Line']
            choice = "SKIP"; odds = 0.0
            if diff > 0: choice = "OVER"; odds = row['Odds_Over']
            elif diff < 0: choice = "UNDER"; odds = row['Odds_Under']
            
            row.update({'Pred_Mean': round(mu,1), 'Edge': round(diff,1), 'Pick': choice, 'Pick_Odds': odds})
            return row
            
        # Collect Individual Bets
        b_seq = process_seq(pred_vals_seq, row)
        b_naive = process_base(p_naive, s_naive, row)
        b_lr = process_base(p_lr, lr_stds, row)
        b_xgb = process_base(p_xgb, xgb_stds, row)
        
        seq_bets.append(b_seq)
        naive_bets.append(b_naive)
        lr_bets.append(b_lr)
        xgb_bets.append(b_xgb)
        
        # --- Hybrid Consensus Logic ---
        # Strategy: All 4 models must agree on Pick (OVER or UNDER)
        # Sequence model acts as the "Base" for output values (Mean/EV), but others validate direction.
        
        picks = [b_seq['Pick'], b_naive['Pick'], b_lr['Pick'], b_xgb['Pick']]
        
        consensus = "SKIP"
        if all(p == "OVER" for p in picks):
            consensus = "OVER"
        elif all(p == "UNDER" for p in picks):
            consensus = "UNDER"
            
        if consensus != "SKIP":
            # Valid Hybrid Bet
            # Use columns from Sequence model as main info, but mark as Hybrid
            # Or maybe average predictions? 
            # User said "Opinions consistent" - usually implies voting.
            # I will use Sequence Model's confidence/EV but only emit if consensus exists.
            h_bet = b_seq.copy()
            h_bet['Method_Vote'] = "4/4"
            hybrid_bets.append(h_bet)
        
    # Save All
    out_map = [
        (seq_bets, OUTPUT_FILE),
        (naive_bets, 'naive_bet.csv'),
        (lr_bets, 'linearRegression_bet.csv'),
        (xgb_bets, 'xgBoost_bet.csv'),
        (hybrid_bets, 'hybrid_bet.csv')
    ]
    
    for data, fname in out_map:
        if not data: 
             # Create empty DF to prevent errors in subsequent scripts if they expect file
             df = pd.DataFrame(columns=['Date','Player_Name','Target','Line','Pick'])
        else:
             df = pd.DataFrame(data)
        
        # Clean cols
        keep = ['Date', 'Player_Name', 'Target', 'Line', 'Odds_Over', 'Odds_Under', 'Pred_Mean', 'Pick', 'Pick_EV', 'Pick_Odds', 'FairOdds_Over', 'FairOdds_Under', 'Method_Vote']
        df = df[[c for c in keep if c in df.columns]]
        
        df.to_csv(os.path.join(ROOT_DIR, fname), index=False)
        
    print(f"Done. Saved bets.csv + 3 baselines + hybrid_bet.csv.")

if __name__ == "__main__":
    main()
