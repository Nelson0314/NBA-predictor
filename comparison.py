import pandas as pd
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from seqModel import createSequences, NbaTransformer
from sklearn.preprocessing import StandardScaler
import torch

def get_target_indices(featureCols, targetCols):
    """
    Find the indices of target columns within the feature columns list.
    """
    indices = []
    for target in targetCols:
        try:
            idx = featureCols.index(target)
            indices.append(idx)
        except ValueError:
            print(f"Warning: Target {target} not found in featureCols. Naive baseline might be incorrect.")
            indices.append(-1)
    return indices

def train_baselines_and_compare(config):
    print("\n" + "="*50)
    print("STARTING BASELINE COMPARISON & REPORTING")
    print("="*50)

    # 1. Load Data (Using multiModel logic for consistency with best model)
    print("Loading data for baselines (using multiModel loader)...")
    from multiModel import loadAndPreprocessData
    
    # multiModel requires: gamesPath, shotsPath, teamsPath, seqLength
    # We retrieve these from config
    gamesPath = config.get('gamesPath', 'dataset/games.csv')
    shotsPath = config.get('shotsPath', 'dataset/shots.csv')
    teamsPath = config.get('teamsPath', 'dataset/teams.csv')
    seqLength = config['seqLength']
    
    # multiModel returns: gamesData, shotsGrouped, featureCols, targetCols
    gamesData, shotsGrouped, featureCols, targetCols = loadAndPreprocessData(
        gamesPath, shotsPath, teamsPath, seqLength
    )

    # 2. Split Data
    trainData = gamesData[gamesData['SEASON_ID'].isin(config['trainSeasons'])].copy()
    testData = gamesData[gamesData['SEASON_ID'].isin(config['testSeasons'])].copy()

    if len(testData) == 0:
        print("Warning: No test data found for comparison. Skipping.")
        return

    # 3. Create Sequences
    print("Generating sequences for baselines...")
    xTrain, yTrain = createSequences(trainData, seqLength, featureCols, targetCols)
    xTest, yTest = createSequences(testData, seqLength, featureCols, targetCols)

    # 4. Preparing Data for Classical Models (Flattening)
    # xTrain shape: (N, seqLength, nFeatures) -> (N, seqLength * nFeatures)
    N_train, S, F = xTrain.shape
    xTrainFlat = xTrain.reshape(N_train, S * F)
    
    N_test, _, _ = xTest.shape
    xTestFlat = xTest.reshape(N_test, S * F)

    # 5. Initialize Results Dictionary
    results = {}

    # Define Metrics
    def calculate_metrics(y_true, y_pred, model_name):
        rmse_per_target = {}
        mse_per_target = {}
        for i, target in enumerate(targetCols):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            rmse_per_target[target] = rmse
            mse_per_target[target] = mse
        
        results[model_name] = {
            'rmse': rmse_per_target,
            'mse': mse_per_target,
            'preds': y_pred  # Store predictions for volatility analysis
        }
        print(f"[{model_name}] Test RMSE: " + ", ".join([f"{k}={v:.4f}" for k,v in rmse_per_target.items()]))

    # --- Model 1: Naive (Last 10 Avg) ---
    print("\nEvaluating Naive Baseline (Average of input window)...")
    target_indices = get_target_indices(featureCols, targetCols)
    
    # xTest: (N, S, F)
    # We take the mean across the sequence dimension (axis 1) for the target features
    yPredNaive = np.zeros_like(yTest)
    
    for i, target_idx in enumerate(target_indices):
        if target_idx != -1:
            # Calculate mean of this feature across the sequence window
            # xTest[:, :, target_idx] is (N, S)
            yPredNaive[:, i] = np.mean(xTest[:, :, target_idx], axis=1)
    
    calculate_metrics(yTest, yPredNaive, 'Naive (Last 10 Avg)')

    # --- Model 2: Linear Regression ---
    print("\nTraining Linear Regression (Scaled)...")
    lr = LinearRegression()
    scaler_lr = StandardScaler()
    xTrainFlatScaled = scaler_lr.fit_transform(xTrainFlat)
    xTestFlatScaled = scaler_lr.transform(xTestFlat)
    
    lr.fit(xTrainFlatScaled, yTrain)
    yPredLR = lr.predict(xTestFlatScaled)
    # Force non-negative
    yPredLR = np.maximum(yPredLR, 0)
    calculate_metrics(yTest, yPredLR, 'Linear Regression')

    # --- Model 3: Random Forest ---
    print("\nTraining Random Forest (n_estimators=50)...") # Reduced estimators for speed
    rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=config['seed'])
    rf.fit(xTrainFlat, yTrain)
    yPredRF = rf.predict(xTestFlat)
    calculate_metrics(yTest, yPredRF, 'Random Forest')

    # --- Model 4: XGBoost ---
    print("\nTraining XGBoost (Tuned)...")
    xgb = XGBRegressor(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=4, 
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1, 
        random_state=config['seed']
    )
    xgb.fit(xTrainFlat, yTrain)
    yPredXGB = xgb.predict(xTestFlat)
    calculate_metrics(yTest, yPredXGB, 'XGBoost')
    
    # --- Model 5: Hybrid House (Ensemble) ---
    print("\nCalc Hybrid House (0.4*LR + 0.45*XGB + 0.15*Naive)...")
    # Weights from tune_house.py
    yPredHybrid = (0.40 * yPredLR) + (0.45 * yPredXGB) + (0.15 * yPredNaive)
    calculate_metrics(yTest, yPredHybrid, 'Hybrid House')

    # 6. Gather Deep Learning Results from savedModels (Pick BEST of each type)
    print("\nGathering Deep Learning Results from 'savedModels'...")
    dl_configs = glob.glob(os.path.join(config['saveDir'], '*', 'config.json'))
    
    # Store all found models by type
    found_models = {
        'multimodal': [],
        'sequence': [],
        'cnn': []
    }
    
    for conf_path in dl_configs:
        try:
            with open(conf_path, 'r') as f:
                c = json.load(f)
            
            folder_name = os.path.basename(os.path.dirname(conf_path))
            
            # Determine type
            m_type = None
            if "multimodal" in folder_name.lower():
                m_type = 'multimodal'
            elif "seq" in folder_name.lower():
                m_type = 'sequence'
            elif "cnn" in folder_name.lower():
                m_type = 'cnn'
            
            if m_type and 'valid_mse' in c and 'test_rmse' in c:
                found_models[m_type].append({
                    'name': folder_name,
                    'config': c,
                    'valid_mse': c['valid_mse']
                })
                
        except Exception as e:
            print(f"Error reading {conf_path}: {e}")

    # Select best of each type
    for m_type, candidates in found_models.items():
        if not candidates:
            continue
            
        # Sort by validation MSE (ascending)
        candidates.sort(key=lambda x: x['valid_mse'])
        best = candidates[0]
        
        display_name = f"Best-{m_type.capitalize()}"
        print(f"Selected {display_name}: {best['name']} (Val MSE: {best['valid_mse']:.4f})")
        
        results[display_name] = {'rmse': best['config']['test_rmse']}
        
        # --- Attempt to load and predict for Volatility Analysis ---
        # We only support 'sequence' model inference here effectively for now
        # because Multimodal requires heavy image loading/processing which is not set up in this script yet.
        if m_type == 'sequence':
            try:
                print(f"Loading {display_name} for inference...")
                
                # 1. Prepare Scalers (Must match training logic!)
                # We fit on xTrain (Raw)
                scalerX = StandardScaler()
                xTrainReshaped = xTrain.reshape(-1, len(featureCols))
                scalerX.fit(xTrainReshaped)
                
                scalerY = StandardScaler()
                scalerY.fit(yTrain)
                
                # 2. Transform Test Data
                xTestReshaped = xTest.reshape(-1, len(featureCols))
                xTestScaled = scalerX.transform(xTestReshaped).reshape(xTest.shape)
                
                # 3. Load Model
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                cfg = best['config']
                
                model = NbaTransformer(
                    inputDim=len(featureCols),
                    dModel=cfg['dModel'],
                    nHead=cfg['nHead'],
                    numLayers=cfg['numLayers'],
                    outputDim=len(targetCols),
                    dropout=cfg.get('dropout', 0.1)
                ).to(device)
                
                ckpt_path = os.path.join(saveDir, best['name'], 'model.ckpt')
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                model.eval()
                
                # 4. Predict
                with torch.no_grad():
                    inputTensor = torch.FloatTensor(xTestScaled).to(device)
                    # Batch prediction (split if too large, but 800 is fine)
                    predScaled = model(inputTensor).cpu().numpy()
                    
                    # 5. Inverse Transform
                    yPredDL = scalerY.inverse_transform(predScaled)
                    
                    # Store
                    results[display_name]['preds'] = yPredDL
                    print(f"  >>> Inference successful. Stored predictions for {display_name}.")
                    
            except Exception as e:
                print(f"  >>> Error running inference for {display_name}: {e}")

    # 7. Analyze Volatility
    volatility_results = analyze_volatility(results, yTest, targetCols, config['saveDir'])
    
    # 8. Generate Report & Plots
    generate_report(results, volatility_results, config)


def analyze_volatility(results, y_true, target_cols, saveDir):
    print("\n" + "="*50)
    print("ANALYZING VOLATILITY")
    print("="*50)
    
    vol_results = {}
    window = 5
    
    # Calculate Rolling Std for Actuals (Ground Truth Volatility)
    # y_true shape: (N, 3) [PTS, AST, REB]
    df_true = pd.DataFrame(y_true, columns=target_cols)
    rolling_std_true = df_true.rolling(window=window).std().fillna(0)
    
    # Plot 1: Volatility Tracking (Rolling Std Comparison)
    for i, target in enumerate(target_cols):
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_std_true[target], label='Actual Volatility', color='black', linewidth=2, linestyle='--')
        
        target_vol_metrics = {}
        
        for model_name, data in results.items():
            # Get predictions for this model
            # Re-predict or store predictions? 
            # Wait, predictions are not stored in 'results' dict passed here. 
            # I need to refactor 'train_baselines_and_compare' to store predictions in 'results' 
            # or pass them separately. 
            # CURRENT STATE: 'results' only has {model: {rmse: ..., mse: ...}}
            # I NEED TO MODIFY 'calculate_metrics' in 'train_baselines_and_compare' to store predictions.
            pass
            
    # RE-THINKING: I need to modify `train_baselines_and_compare` first to save predictions in `results`.
    return {} # Placeholder until refactor



def analyze_volatility(results, y_true, target_cols, saveDir):
    print("\n" + "="*50)
    print("ANALYZING VOLATILITY (Rolling Window=5)")
    print("="*50)
    
    vol_metrics = {}
    window = 5
    
    # DataFrame for Actuals
    df_true = pd.DataFrame(y_true, columns=target_cols)
    rolling_std_true = df_true.rolling(window=window).std()
    
    # Prepare Plot 1: Volatility Tracking
    for i, target in enumerate(target_cols):
        plt.figure(figsize=(14, 6))
        
        # Plot Actual
        plt.plot(rolling_std_true[target], label='Actual Volatility (Rolling Std)', color='black', linewidth=2, linestyle='--')
        
        for model_name, data in results.items():
            if 'preds' not in data: continue
            
            y_pred = data['preds']
            df_pred = pd.DataFrame(y_pred, columns=target_cols)
            rolling_std_pred = df_pred.rolling(window=window).std()
            
            # Metric: Volatility Ratio (Predicted Volatility / Actual Volatility)
            # Ideally should be close to 1.0. <1.0 = Too Smooth. >1.0 = Too Noisy.
            mean_vol_true = rolling_std_true[target].mean()
            mean_vol_pred = rolling_std_pred[target].mean()
            
            vol_ratio = mean_vol_pred / mean_vol_true if mean_vol_true > 1e-6 else 0
            
            # Store Metric
            if model_name not in vol_metrics: vol_metrics[model_name] = {}
            vol_metrics[model_name][f'{target}_VolRatio'] = vol_ratio
            
            # Add to Plot
            plt.plot(rolling_std_pred[target], label=f"{model_name} (Ratio={vol_ratio:.2f})", alpha=0.7)
            
        plt.title(f"{target} - Volatility Analysis (Rolling Std, Window={window})")
        plt.xlabel("Game Index")
        plt.ylabel("Rolling Standard Deviation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(saveDir, f'volatility_tracking_{target}.png')
        plt.savefig(plot_path)
        print(f"Saved Volatility Plot: {plot_path}")
        plt.close()
        
    return vol_metrics

def generate_report(results, vol_results, config):
    print("\nGenerating Report and Charts...")
    saveDir = config['saveDir']
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # --- Bar Chart (RMSE) ---
    targets = ['PTS', 'AST', 'REB']
    models = list(results.keys())
    models.sort() 

    # Prepare data for plotting
    plot_data = {t: [] for t in targets}
    
    for model in models:
        for t in targets:
            val = results[model]['rmse'].get(t, 0)
            plot_data[t].append(val)

    x = np.arange(len(models))
    width = 0.25 
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(12, 6))

    for attribute, measurement in plot_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.2f')
        multiplier += 1

    ax.set_ylabel('RMSE (Lower is Better)')
    ax.set_title('Model Comparison on 2024-25 Test Set')
    ax.set_xticks(x + width, models)
    ax.legend(loc='upper left', ncols=3)
    
    plot_path = os.path.join(saveDir, 'comparison_chart.png')
    plt.savefig(plot_path)
    print(f"Chart saved to: {plot_path}")

    # --- Markdown Report ---
    report_path = os.path.join(saveDir, 'comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# NBA Model Comparison Report\n\n")
        f.write(f"**Test Season**: {config['testSeasons']}\n")
        f.write(f"**Sequence Length**: {config['seqLength']}\n\n")
        
        f.write("## 1. Metrics Comparison (RMSE)\n")
        f.write("| Model | PTS | AST | REB |\n")
        f.write("|---|---|---|---|\n")
        
        for model in models:
            r = results[model]['rmse']
            pts = f"{r.get('PTS', 0):.4f}"
            ast = f"{r.get('AST', 0):.4f}"
            reb = f"{r.get('REB', 0):.4f}"
            f.write(f"| {model} | {pts} | {ast} | {reb} |\n")
        
        f.write("\n## 2. Volatility Analysis (Variance Capture)\n")
        f.write("This section measures how well the model captures the natural variance of player stats.\n")
        f.write("- **Volatility Ratio**: `Avg(Pred_Std) / Avg(Actual_Std)`\n")
        f.write("- **Target**: `1.0`. \n")
        f.write("- `< 1.0`: Model is too safe/smooth (regressing to mean).\n")
        f.write("- `> 1.0`: Model is too noisy/erratic.\n\n")
        
        f.write("| Model | PTS VolRatio | AST VolRatio | REB VolRatio |\n")
        f.write("|---|---|---|---|\n")
        
        for model in models:
            if model in vol_results:
                v = vol_results[model]
                pts = f"{v.get('PTS_VolRatio', 0):.2f}"
                ast = f"{v.get('AST_VolRatio', 0):.2f}"
                reb = f"{v.get('REB_VolRatio', 0):.2f}"
                f.write(f"| {model} | {pts} | {ast} | {reb} |\n")
            else:
                f.write(f"| {model} | N/A | N/A | N/A |\n")

        f.write("\n### Volatility Charts\n")
        f.write("![PTS Volatility](volatility_tracking_PTS.png)\n")
        f.write("![AST Volatility](volatility_tracking_AST.png)\n")
        f.write("![REB Volatility](volatility_tracking_REB.png)\n")
        
        f.write("\n## 3. Visualization\n")
        f.write("![Comparison Chart](comparison_chart.png)\n")
        
        f.write("\n## 4. Configuration & Parameters\n")
        f.write("### Baseline Parameters\n")
        f.write("- **XGBoost**: n_estimators=100, learning_rate=0.1\n")
        f.write("- **RandomForest**: n_estimators=50\n")
        f.write("- **LinearRegression**: Default sklearn\n")
        f.write("- **Naive**: Average of features in input window (Last 10 games)\n")
        
        f.write("\n### Deep Learning Config\n")
        f.write("Only showing generic config used for this run:\n")
        f.write("```json\n")
        # Filter out non-serializable objects from config if any
        serializable_config = {k:v for k,v in config.items() if isinstance(v, (int, float, str, list, bool))}
        json.dump(serializable_config, f, indent=4)
        f.write("\n```\n")

    print(f"Report saved to: {report_path}")


def analyze_specific_player(results, config, target_player_name="Stephen Curry", target_season_id=22025):
    print(f"\n" + "="*50)
    print(f"ANALYZING SPECIFIC PLAYER: {target_player_name} ({target_season_id})")
    print("="*50)
    
    # 1. Load Data Config
    from multiModel import loadAndPreprocessData
    gamesPath = config.get('gamesPath', 'dataset/games.csv')
    shotsPath = config.get('shotsPath', 'dataset/shots.csv')
    teamsPath = config.get('teamsPath', 'dataset/teams.csv')
    seqLength = config['seqLength']
    
    print("Loading local dataset metadata...")
    gamesData, shotsGrouped, featureCols, targetCols = loadAndPreprocessData(
        gamesPath, shotsPath, teamsPath, seqLength
    )
    
    # 2. Try to find player in Local Data
    playerData = pd.DataFrame()
    found_locally = False
    
    if 'Player_Name' in gamesData.columns:
        playerData = gamesData[(gamesData['Player_Name'] == target_player_name) & (gamesData['SEASON_ID'] == target_season_id)].copy()
    
    if len(playerData) > 0:
        print(f"Found {len(playerData)} games locally for {target_player_name}.")
        found_locally = True
    else:
        print(f"No local games found for {target_player_name} in {target_season_id}.")
        print("Attempting to fetch LIVE data via nba_api...")
        try:
            from nba_api.stats.endpoints import playergamelog
            from nba_api.stats.static import players
            import time
            
            # Find Player ID
            nba_players = players.get_players()
            curry_entry = next((p for p in nba_players if p['full_name'] == target_player_name), None)
            
            if not curry_entry:
                print(f"Error: Could not find ID for {target_player_name} in nba_api.")
                return
            
            pid = curry_entry['id']
            print(f"Fetched Player ID: {pid}")
            
            # Fetch Game Log for 2025-26 Season
            season_str = "2025-26" 
            print(f"Fetching Game Log for Season: {season_str}")
            
            gamelog = playergamelog.PlayerGameLog(player_id=pid, season=season_str)
            df_live = gamelog.get_data_frames()[0]
            
            if len(df_live) == 0:
                print("No live games found.")
                return
                
            print(f"Fetched {len(df_live)} games from NBA API.")
            
            # 3. Process Live Data
            df_live['GAME_DATE'] = pd.to_datetime(df_live['GAME_DATE'])
            df_live['SEASON_ID'] = target_season_id 
            df_live['Player_ID'] = pid
            
            for col in featureCols:
                if col not in df_live.columns:
                    df_live[col] = 0.0 
                    
            def parse_matchup(m):
                if ' vs. ' in m: return m.split(' vs. ')
                if ' @ ' in m: return m.split(' @ ')
                return [None, None]

            matchups = df_live['MATCHUP'].apply(parse_matchup)
            df_live['TEAM_ABBREVIATION'] = [x[0] for x in matchups]
            df_live['OPPONENT_ABBREVIATION'] = [x[1] for x in matchups]
            
            df_live = df_live.sort_values(by='GAME_DATE').reset_index(drop=True)
            
            # Player Rolling
            p_cols = ['PTS', 'AST', 'REB']
            for c in p_cols:
                new_c = f'PLAYER_AVG_{c}'
                df_live[new_c] = df_live[c].expanding().mean().shift(1).fillna(0)
                
            df_live['DAYS_SINCE_LAST_GAME'] = df_live['GAME_DATE'].diff().dt.days.fillna(7)
            
            playerData = df_live
            playerData['GAME_ID'] = playerData['Game_ID']
            
        except Exception as e:
            print(f"Failed to fetch live data: {e}")
            return

    # Check Columns overlap
    missing_cols = [c for c in featureCols if c not in playerData.columns]
    if missing_cols:
        for c in missing_cols:
            playerData[c] = 0.0

    print("Data prepared. Generating sequences...")
    playerData = playerData.sort_values(by='GAME_DATE')
    
    # 3. Create Sequences
    xTest, yTest = createSequences(playerData, seqLength, featureCols, targetCols)
    
    if len(xTest) == 0:
        print("Not enough games to form a sequence (Need > seqLength).")
        return
        
    dates = playerData['GAME_DATE'].iloc[seqLength:].values
    
    # 4. Prepare Data Layout
    N, S, F = xTest.shape
    xTestFlat = xTest.reshape(N, S * F)
    
    # 5. Run Predictions
    player_results = {'Actual': yTest}
    
    # --- Baselines ---
    trainData = gamesData[gamesData['SEASON_ID'].isin(config['trainSeasons'])].copy()
    xTrain, yTrain = createSequences(trainData, seqLength, featureCols, targetCols)
    xTrainFlat = xTrain.reshape(xTrain.shape[0], -1)
    
    print("Training Baselines...")
    print("Training Baselines...")
    # Linear Regression (Scaled)
    lr = LinearRegression()
    scaler_lr = StandardScaler()
    xTrainFlatScaled = scaler_lr.fit_transform(xTrainFlat)
    xTestFlatScaled = scaler_lr.transform(xTestFlat)
    
    lr.fit(xTrainFlatScaled, yTrain)
    pred_lr = np.maximum(lr.predict(xTestFlatScaled), 0)
    player_results['Linear Regression'] = pred_lr
    
    # XGBoost (Tuned)
    xgb = XGBRegressor(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=4, 
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1, 
        random_state=42
    )
    xgb.fit(xTrainFlat, yTrain)
    pred_xgb = xgb.predict(xTestFlat)
    player_results['XGBoost'] = pred_xgb
    
    # Naive (Needs calculation if not already there, assuming logic matches above)
    # We need naive for Hybrid. 
    # Logic: Last 10 games average from input
    # xTest is (N, seqLength, Features). We need specific features.
    # We can approximate "Naive" using the player's last known rolling avg from input features if available,
    # or just calc mean of input window like in training function.
    
    target_indices = get_target_indices(featureCols, targetCols)
    pred_naive = np.zeros_like(pred_lr)
    for i, idx in enumerate(target_indices):
        if idx != -1:
             pred_naive[:, i] = np.mean(xTest[:, :, idx], axis=1)
             
    # Hybrid House
    player_results['Hybrid House'] = (0.40 * pred_lr) + (0.45 * pred_xgb) + (0.15 * pred_naive)
    
    # --- Multimodal Model ---
    dl_configs = glob.glob(os.path.join(config['saveDir'], '*', 'config.json'))
    best_multi_mse = float('inf')
    best_multi_path = None
    
    for c_path in dl_configs:
        if 'multimodal' in c_path.lower():
            with open(c_path) as f: cfg = json.load(f)
            if cfg.get('valid_mse', float('inf')) < best_multi_mse:
                best_multi_mse = cfg['valid_mse']
                best_multi_path = os.path.dirname(c_path)
    
    if best_multi_path:
        print(f"Loading Best Multimodal Model from: {best_multi_path}")
        from multiModel import NbaMultimodal, createMultimodalSequences
        from graphModel import generateHeatmap # Import Heatmap Generator
        from nba_api.stats.endpoints import shotchartdetail
        import torch
        import time
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        with open(os.path.join(best_multi_path, 'config.json')) as f: modelval_config = json.load(f)
        
        model = NbaMultimodal(
            numStatFeatures=len(featureCols),
            seqLength=modelval_config.get('seqLength', 10),
            outputDim=len(targetCols),
            cnnEmbedDim=modelval_config.get('cnnEmbedDim', 64),
            statEmbedDim=modelval_config.get('statEmbedDim', 128),
            dModel=modelval_config.get('dModel', 128),
            nHead=modelval_config.get('nHead', 4),
            numLayers=modelval_config.get('numLayers', 2),
            dropout=modelval_config.get('dropout', 0.1)
        ).to(device)
        
        try:
            model.load_state_dict(torch.load(os.path.join(best_multi_path, 'model.ckpt')))
            model.eval()
            
            # Create Sequences (Game IDs)
            dummyShots = {} 
            _, xGameTest, xStatTest, _ = createMultimodalSequences(
                playerData, dummyShots, seqLength, featureCols, targetCols
            )
            
            # Generate Live Heatmaps
            # xGameTest: (N, S) - GameIDs
            print(f"Generating LIVE Heatmaps for {len(xGameTest)} sequences (seq_len={seqLength})...")
            
            all_imgs = []
            
            # Optimization: Cache heatmaps for GameIDs we've already fetched in this session
            session_heatmap_cache = {} 
            
            pid = playerData['Player_ID'].iloc[0]
            
            for i in range(len(xGameTest)):
                seq_imgs = []
                for j in range(seqLength):
                    gid = xGameTest[i][j]
                    
                    if gid in session_heatmap_cache:
                        heatmap = session_heatmap_cache[gid]
                    else:
                        print(f"  Fetching Shot Chart for Game: {gid}")
                        try:
                            # Fetch Shot Chart
                            response = shotchartdetail.ShotChartDetail(
                                team_id=0,
                                player_id=pid,
                                game_id=gid,
                                season_nullable='2025-26',
                                context_measure_simple='FGA'
                            )
                            shots_df = response.get_data_frames()[0]
                            time.sleep(0.3) # Rate limit politeness
                            
                            # Generate Heatmap (using graphModel logic)
                            # graphModel.generateHeatmap expects DataFrame with LOC_X, LOC_Y, SHOT_MADE_FLAG
                            heatmap = generateHeatmap(shots_df) # Returns FloatTensor (2, 50, 50)
                            
                        except Exception as e:
                            print(f"    Error fetching/generating heatmap for {gid}: {e}")
                            heatmap = torch.zeros((2, 50, 50), dtype=torch.float32)
                        
                        session_heatmap_cache[gid] = heatmap
                        
                    seq_imgs.append(heatmap.numpy())
                
                all_imgs.append(seq_imgs)
                
            # Convert to Tensor (N, S, 2, 50, 50)
            xImgTensor = torch.FloatTensor(np.array(all_imgs)).to(device)
            
            with torch.no_grad():
                 scalerX = StandardScaler()
                 scalerX.fit(xTrain.reshape(-1, len(featureCols)))
                 scalerY = MinMaxScaler(feature_range=(0,1))
                 scalerY.fit(yTrain)
                 
                 xStatTestScaled = scalerX.transform(xStatTest.reshape(-1, len(featureCols))).reshape(xStatTest.shape)
                 xStatTensor = torch.FloatTensor(xStatTestScaled).to(device)
                 
                 preds = model(xImgTensor, xStatTensor)
                 predsOriginal = scalerY.inverse_transform(preds.cpu().numpy())
                 player_results['Multimodal (Live Heatmaps)'] = np.maximum(predsOriginal, 0)
                 
        except Exception as e:
            print(f"Error running Multimodal model on live data: {e}")
            import traceback
            traceback.print_exc()

    # 6. Plotting
    print("Generating Plots...")
    for i, target in enumerate(targetCols):
        plt.figure(figsize=(14, 6))
        
        time_labels = [str(d).split('T')[0] for d in dates]
        x_indices = range(len(dates))
        
        plt.plot(x_indices, yTest[:, i], label='Actual', color='black', linewidth=2, linestyle='--')
        
        for model_name, preds in player_results.items():
            if model_name == 'Actual': continue
            plt.plot(x_indices, preds[:, i], label=model_name, marker='o', markersize=4, alpha=0.7)
            
        plt.title(f"{target_player_name} {target_season_id} (2025-26) - {target} Prediction")
        plt.xlabel("Game Date")
        plt.ylabel(target)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(x_indices, time_labels, rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(config['saveDir'], f"{target_player_name}_{target}_2025.png"))
        print(f"Saved plot: {target_player_name}_{target}_2025.png")

if __name__ == "__main__":
    # Default config for testing
    config = {
        'seed': 42,
        'seqLength': 10,
        'saveDir': 'savedModels',
        'gamesPath': 'dataset/games.csv', 
        'shotsPath': 'dataset/shots.csv',
        'teamsPath': 'dataset/teams.csv',
        'trainSeasons': [22016, 22017, 22018, 22019, 22020, 22021, 22022],
        'valSeasons': [22023],
        'testSeasons': [22024]
    }
    # Run Baseline Comparison
    train_baselines_and_compare(config)
    
    # Run Specific Player Analysis (Live Data)
    analyze_specific_player(None, config, "Stephen Curry", 22025) # 22025 = 2025-26

