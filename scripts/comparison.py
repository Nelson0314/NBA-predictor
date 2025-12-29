import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR, SAVED_MODELS_DIR, HEATMAP_DIR
from src.multiModel import loadAndPreprocessData, createMultimodalSequences, preloadHeatmaps, MultimodalDataset, NbaMultimodal
from src.seqModel import NbaTransformer, NbaSequenceDataset

# Config
SEQ_LENGTH = 7
BATCH_SIZE = 64
TRAIN_SEASONS = [22016, 22017, 22018, 22019, 22020, 22021, 22022]
TEST_SEASONS = [22024]
TARGET_COLS = ['PTS', 'AST', 'REB']

def get_sequences_and_features():
    print("Loading Data...")
    gamesPath = os.path.join(DATA_DIR, 'games.csv')
    shotsPath = os.path.join(DATA_DIR, 'shots.csv')
    teamsPath = os.path.join(DATA_DIR, 'teams.csv')
    
    gamesData, _, featureCols, targetCols = loadAndPreprocessData(gamesPath, shotsPath, teamsPath, SEQ_LENGTH)
    
    # Fix GAME_ID for Heatmap Lookup
    if 'GAME_ID' in gamesData.columns:
        gamesData['GAME_ID'] = pd.to_numeric(gamesData['GAME_ID'], errors='coerce').fillna(0).astype('int64').astype(str).str.zfill(10)
        print(f"Patched GAME_ID in comparison_dl.py. Sample: {gamesData['GAME_ID'].iloc[0]}")
    
    heatmapCache = preloadHeatmaps(HEATMAP_DIR)
    
    trainData = gamesData[gamesData['SEASON_ID'].isin(TRAIN_SEASONS)]
    testData = gamesData[gamesData['SEASON_ID'].isin(TEST_SEASONS)]
    
    print("Generating Sequences...")
    # For Baselines (Features + Vis PCA)
    pTrain, gTrain, xTrain, yTrain = createMultimodalSequences(trainData, None, SEQ_LENGTH, featureCols, TARGET_COLS)
    pTest, gTest, xTest, yTest = createMultimodalSequences(testData, None, SEQ_LENGTH, featureCols, TARGET_COLS)
    
    # Create DataLoaders
    trainDataset = MultimodalDataset(pTrain, gTrain, xTrain, heatmapCache, yTrain)
    testDataset = MultimodalDataset(pTest, gTest, xTest, heatmapCache, yTest)
    
    # Multimodal Loader for DL
    dlTrainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=False)
    dlTestLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Helper to extract flattened features for Baseline
    def extract_flat(loader):
        vis_list, stat_list, y_list = [], [], []
        with torch.no_grad():
            for imgs, stats, targets in tqdm(loader, desc="Extracting Flat Features"):
                avgImg = torch.mean(imgs, dim=1) # (B, 2, 50, 50)
                flat_vis = avgImg.numpy().reshape(avgImg.shape[0], -1) 
                flat_stat = stats.view(stats.shape[0], -1).numpy()
                vis_list.append(flat_vis)
                stat_list.append(flat_stat)
                y_list.append(targets.numpy())
        return np.vstack(vis_list), np.vstack(stat_list), np.vstack(y_list)

    print("Extracting Train Features for Baselines...")
    X_train_vis_raw, X_train_stat, y_train = extract_flat(dlTrainLoader)
    print("Extracting Test Features for Baselines...")
    X_test_vis_raw, X_test_stat, y_test = extract_flat(dlTestLoader)
    
    return X_train_vis_raw, X_train_stat, y_train, X_test_vis_raw, X_test_stat, y_test, dlTestLoader, featureCols, xTrain, xTest, pTest, gTest, yTest, heatmapCache

def evaluate_dl_model(name, model_dir, testLoader, device, featureCols, scalerY=None):
    try:
        config_path = os.path.join(model_dir, 'config.json')
        ckpt_path = os.path.join(model_dir, 'model.ckpt')
        
        if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
            return f"{name}: Model not found", 999.0
            
        with open(config_path, 'r') as f:
            conf = json.load(f)
            
        # Determine Feature Count
        # Prefer config, fallback to passed featureCols (actual data dim)
        conf_feats = conf.get('featureCols', [])
        num_feats = len(conf_feats) if conf_feats else len(featureCols)
        
        # Determine Model Type
        if "multi" in name.lower() or "multimodal" in name.lower():
            model = NbaMultimodal(
                numStatFeatures=num_feats,
                 seqLength=conf.get('seqLength', 7),
                 outputDim=9, 
                 dModel=conf.get('dModel', 128),
                 statEmbedDim=conf.get('statEmbedDim', 128),
                 numLayers=conf.get('numLayers', 3),
                 nHead=conf.get('nHead', 4),
                 cnnEmbedDim=conf.get('cnnEmbedDim', 64)
            )
        else:
            # Seq
             model = NbaTransformer(
                inputDim=num_feats,
                dModel=conf.get('dModel', 64),
                nHead=conf.get('nHead', 4),
                numLayers=conf.get('numLayers', 3), # Error suggested deep model? Seq MAE was 138, but it loaded.
                # So Seq numLayers=2 was fine? 
                # Let's trust config for Seq, or default to 3 if user used 3.
                outputDim=3, 
                statEmbedDim=conf.get('statEmbedDim', 128)
            )

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        
        preds = []
        targets = []
        
        with torch.no_grad():
            for imgs, stats, y in testLoader:
                imgs, stats = imgs.to(device), stats.to(device)
                
                if "multi" in name.lower():
                    out = model(imgs, stats)
                else:
                    out = model(stats) # Seq only takes stats
                
                # Out shape: (B, T, 3, 3) or (B, T, 9)
                if out.dim() == 2:
                    out = out.view(out.size(0), 3, 3)
                
                # Take P50 (Index 1)
                p50 = out[:, :, 1]
                preds.append(p50.cpu().numpy())
                targets.append(y[:, :3].numpy())
                
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0) # RAW targets

        # Inverse Transform Preds ONLY (Assume model output is scaled)
        if scalerY:
             preds = scalerY.inverse_transform(preds)
             # targets are ALREADY raw (from loader), DO NOT transform.
             
        # Calc Metrics
        maes = []
        stds = []
        for i in range(3):
            loss = np.abs(targets[:, i] - preds[:, i])
            maes.append(np.mean(loss))
            stds.append(np.std(loss))
            
        avg_mae = np.mean(maes)
        avg_std = np.mean(stds)
        
        return_str = f"{name:<30} | {maes[0]:<10.4f} | {maes[1]:<10.4f} | {maes[2]:<10.4f} | {avg_mae:<10.4f} | {avg_std:<10.4f}"
        return return_str, avg_mae
        
    except Exception as e:
        return f"Error evaluating {name}: {str(e)}", 999.0

def train_and_evaluate():
    # 1. Get Data
    X_train_vis_raw, X_train_stat, y_train, X_test_vis_raw, X_test_stat, y_test, dlTestLoader, featureCols, xTrain, xTest, pTest, gTest, yTestRaw, heatmapCache = get_sequences_and_features()

    # Create TWO scalers: Multi uses MinMaxScaler, Seq uses StandardScaler
    scalerY_mm = MinMaxScaler(feature_range=(0, 1))
    scalerY_mm.fit(y_train[:, :3])
    
    scalerY_std = StandardScaler()
    scalerY_std.fit(y_train[:, :3]) 
    # Wait, baselines predict unscaled targets usually? 
    # No, baseline.py usually feeds scalerY inverse transforms?
    # Original baseline.py didn't use scalerY for baselines explicitly unless implemented.
    # Checking original code: y_train is from dataset.
    # Dataset loads targets. Targets in loadAndPreprocessData are usually NOT scaled unless scaler passed.
    # loadAndPreprocessData returns scaler? No.
    # So y_train is raw.
    # But DL models allow training on scaled data.
    # train.py scales targets.
    # If saved DL models expect SCALED input/target, we have a mismatch!
    # Saved models were trained with Scaler.
    # We MUST scale inputs for DL models. 
    # And inverse transform outputs.
    
    # ... (Reusing existing baseline implementation for completeness) ...
    # To keep it simple, I will focus on ADDING the DL rows.
    # I will rely on 'savedModels_conf' scaler if available? 
    # Or just re-fit scalerY on trainData (which matches training logic).
    
    print("Fitting PCA on Visual Features...")
    scaler_vis = StandardScaler()
    X_train_vis_scaled = scaler_vis.fit_transform(X_train_vis_raw)
    X_test_vis_scaled = scaler_vis.transform(X_test_vis_raw)
    
    pca = PCA(n_components=50)
    X_train_vis_pca = pca.fit_transform(X_train_vis_scaled)
    X_test_vis_pca = pca.transform(X_test_vis_scaled)
    print(f"PCA Explained Variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    # Define Baselines
    X_train_no_cnn = X_train_stat
    X_test_no_cnn = X_test_stat
    X_train_with_cnn = np.hstack([X_train_stat, X_train_vis_pca])
    X_test_with_cnn = np.hstack([X_test_stat, X_test_vis_pca])
    
    configs = [
        ("Naive (Mean)", "NAIVE", None),
        ("Linear Regression (No CNN)", "NO_CNN", LinearRegression()),
        ("XGBoost (No CNN)", "NO_CNN", XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1)),
        ("Linear Regression (With CNN)", "WITH_CNN", LinearRegression()),
        ("XGBoost (With CNN)", "WITH_CNN", XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1))
    ]

    print("\n" + "="*100)
    print(f"{'Model':<30} | {'PTS MAE':<10} | {'AST MAE':<10} | {'REB MAE':<10} | {'AVG MAE':<10} | {'AVG STD':<10}")
    print("-" * 100)
    
    report_lines = []
    header = f"{'Model':<30} | {'PTS MAE':<10} | {'AST MAE':<10} | {'REB MAE':<10} | {'AVG MAE':<10} | {'AVG STD':<10}"
    report_lines.append("="*100)
    report_lines.append(header)
    report_lines.append("-" * 100)

    # 1. Run Baselines
    for name, feat_type, model in configs:
        preds = np.zeros_like(y_test)
        if feat_type == "NAIVE":
            train_means = np.mean(y_train, axis=0)
            preds = np.tile(train_means, (len(y_test), 1))
        else:
            X_tr = X_train_no_cnn if feat_type == "NO_CNN" else X_train_with_cnn
            X_te = X_test_no_cnn if feat_type == "NO_CNN" else X_test_with_cnn
            if isinstance(model, LinearRegression):
                model.fit(X_tr, y_train)
                preds = model.predict(X_te)
            else:
                for i in range(3):
                    model.fit(X_tr, y_train[:, i])
                    preds[:, i] = model.predict(X_te)
        
        maes = []
        stds = []
        for i in range(3):
            loss = np.abs(y_test[:, i] - preds[:, i])
            maes.append(np.mean(loss))
            stds.append(np.std(loss))
        avg_mae = np.mean(maes)
        avg_std = np.mean(stds)
        
        line = f"{name:<30} | {maes[0]:<10.4f} | {maes[1]:<10.4f} | {maes[2]:<10.4f} | {avg_mae:<10.4f} | {avg_std:<10.4f}"
        print(line)
        report_lines.append(line)

    # 2. Run DL Models
    # Need to normalize features similar to training?
    # train.py scales features using StandardScaler per feature col?
    # Actually train.py scales:
    # xScalar = StandardScaler()
    # xTrain = xScalar.fit_transform(xTrain.reshape(-1, xTrain.shape[-1])).reshape(xTrain.shape)
    # We MUST replicate this.
    # The 'dlTestLoader' we created uses 'xTest' from 'createMultimodalSequences'.
    # Does 'createMultimodalSequences' scale? No.
    # So we must scale 'xTrain' and 'xTest' before passing to dlTestLoader?
    # OR we scale inside EVAL function using a fitted scaler?
    # We should fit scaler on xTrain, transform xTest.
    
    # Scale Features for DL
    print("Preparing Data for DL Models (Scaling)...")
    # xTrain shape: (N, T, F)
    N, T, F = xTrain.shape
    xScaler = StandardScaler()
    xTrainFlat = xTrain.reshape(-1, F)
    xTrainScaled = xScaler.fit_transform(xTrainFlat).reshape(N, T, F)
    
    Nt, Tt, Ft = xTest.shape
    xTestFlat = xTest.reshape(-1, Ft)
    xTestScaled = xScaler.transform(xTestFlat).reshape(Nt, Tt, Ft)
    
    # Re-create DL loader with SCALED features
    # Note: Heatmaps (imgs) are 0-1 usually, no scaling needed or handled by BN/CNN?
    # train.py does NOT scale heatmaps.
    
    # Reuse heatmapCache from get_sequences_and_features (no duplicate loading)
    testDatasetScaled = MultimodalDataset(pTest, gTest, xTestScaled, heatmapCache, yTestRaw)
    dlTestLoaderScaled = DataLoader(testDatasetScaled, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (Hardcoded as requested)
    seq_model_dir = os.path.join("savedModels_conf", "seq")
    multi_model_dir = os.path.join("savedModels_conf", "multi")
    
    # Needs to handle Quantile logic (which means outputDim=outputDim*3)
    # The wrapper below handles P50 extraction
    
    # SeqModel uses StandardScaler for Y
    line_seq, _ = evaluate_dl_model("Transformer Seq (Full)", seq_model_dir, dlTestLoaderScaled, device, featureCols, scalerY_std)
    if line_seq:
        print(line_seq)
        report_lines.append(line_seq)
        
    # MultiModel uses MinMaxScaler for Y
    line_multi, _ = evaluate_dl_model("Transformer Multi (Full)", multi_model_dir, dlTestLoaderScaled, device, featureCols, scalerY_mm)
    if line_multi:
        print(line_multi)
        report_lines.append(line_multi)

    print("="*100)
    report_lines.append("="*100)
    
    # Save
    with open("baseline_report_all.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("Report saved to baseline_report_all.txt")

if __name__ == "__main__":
    train_and_evaluate()
