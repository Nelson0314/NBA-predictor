
import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.seqModel import loadAndPreprocessData, createSequences

def train_baselines(seqLength=5):
    print(f"==================================================")
    print(f"Running Baselines (SeqLength={seqLength})")
    print(f"==================================================")
    
    # 1. Load Data
    datasetPath = 'dataset/games.csv'
    gamesData, featureCols, targetCols = loadAndPreprocessData(datasetPath, seqLength)
    
    # 2. Split Data (Same as train.py default)
    trainSeasons = [22016, 22017, 22018, 22019, 22020, 22021, 22022]
    valSeasons = [22023] 
    testSeasons = [22024]
    
    trainData = gamesData[gamesData['SEASON_ID'].isin(trainSeasons)].copy()
    testData = gamesData[gamesData['SEASON_ID'].isin(testSeasons)].copy() # We focus on Test for final report
    
    print(f"Train Records: {len(trainData)}")
    print(f"Test Records:  {len(testData)}")
    
    # 3. Create Sequences
    print("Creating Sequences...")
    # createSequences returns x, y, meta. We only need x, y for baselines.
    xTrain, yTrain, _ = createSequences(trainData, seqLength, featureCols, targetCols)
    xTest, yTest, _ = createSequences(testData, seqLength, featureCols, targetCols)
    
    # 4. Flatten for Traditional Models (N, Seq, Feat) -> (N, Seq*Feat)
    N_tr, S_tr, F_tr = xTrain.shape
    xTrainFlat = xTrain.reshape(N_tr, -1)
    
    N_te, S_te, F_te = xTest.shape
    xTestFlat = xTest.reshape(N_te, -1)
    
    # 5. Scaling (Linear Regression benefits from scaling)
    # User requested defaults, so maybe raw data? 
    # But LR really needs scaling for convergence if SGD, but sklearn use OLS (SVD) so scaling is less critical for solution but good practice.
    # However, to "weaken" it or make it "default", removing scaling is a valid "un-tuning".
    # Let's use raw data for defaults.
    
    # scalerX = StandardScaler()
    # xTrainFlatScaled = scalerX.fit_transform(xTrainFlat)
    # xTestFlatScaled = scalerX.transform(xTestFlat)
    
    # Targets usually don't need scaling for these baselines unless we want to, 
    # but to match train.py raw output interpretation, we'll keep targets raw.
    
    # ==========================================
    # 6. Train & Evaluate Models
    # ==========================================
    
    results = {}
    
    # --- Naive (Window Mean) ---
    print("\n1. Naive (Window Mean)...")
    naive_preds = np.zeros_like(yTest)
    for i, tgt in enumerate(targetCols):
        if tgt in featureCols:
            idx = featureCols.index(tgt)
            # Mean over sequence dimension
            naive_preds[:, i] = np.mean(xTest[:, :, idx], axis=1)
        else:
            # Fallback to global train mean if target not in features (unlikely for auto-regressive)
            naive_preds[:, i] = np.mean(yTrain[:, i])
            
    results['Naive'] = {
        'MAE': mean_absolute_error(yTest, naive_preds),
        'MSE': mean_squared_error(yTest, naive_preds)
    }

    # --- Linear Regression (Default) ---
    print("2. Linear Regression (Default)...")
    lr = LinearRegression()
    lr.fit(xTrainFlat, yTrain) # Train on raw
    lr_preds = lr.predict(xTestFlat)
    
    results['LinearRegression'] = {
        'MAE': mean_absolute_error(yTest, lr_preds),
        'MSE': mean_squared_error(yTest, lr_preds)
    }
    
    # --- XGBoost (Default) ---
    print("3. XGBoost (Default)...")
    # Using params from train.py house_xgb
    xgb = XGBRegressor(n_jobs=-1, random_state=42) # Pure defaults
    
    try:
        xgb.fit(xTrainFlat, yTrain)
        xgb_preds = xgb.predict(xTestFlat)
    except Exception as e:
        print(f"XGBoost fit failed: {e}. Trying MultiOutputRegressor...")
        from sklearn.multioutput import MultiOutputRegressor
        xgb = MultiOutputRegressor(XGBRegressor(n_jobs=-1, random_state=42))
        xgb.fit(xTrainFlat, yTrain)
        xgb_preds = xgb.predict(xTestFlat)

    results['XGBoost'] = {
        'MAE': mean_absolute_error(yTest, xgb_preds),
        'MSE': mean_squared_error(yTest, xgb_preds)
    }
    
    # ==========================================
    # 7. Report Per Target
    # ==========================================
    print("\n" + "="*80)
    print(f"BASELINE REPORT (Test Set 2024-25, Seq={seqLength})")
    print("="*80)
    print(f"{'Model':<12} | {'Metric':<6} | {'PTS':<10} | {'AST':<10} | {'REB':<10}")
    print("-" * 80)
    
    models_preds = {
        'Naive': naive_preds,
        'Linear': lr_preds,
        'XGBoost': xgb_preds
    }
    
    for model_name, preds in models_preds.items():
        # Calculate MAE & Std for each target
        maes = []
        stds = []
        
        for i, tgt in enumerate(targetCols):
            errors = yTest[:, i] - preds[:, i]
            abs_errors = np.abs(errors)
            
            maes.append(np.mean(abs_errors))
            stds.append(np.std(errors))
            
        # Print MAE Row
        print(f"{model_name:<12} | {'MAE':<6} | {maes[0]:<10.4f} | {maes[1]:<10.4f} | {maes[2]:<10.4f}")
        # Print Std Row
        print(f"{'':<12} | {'StdErr':<6} | {stds[0]:<10.4f} | {stds[1]:<10.4f} | {stds[2]:<10.4f}")
        print("-" * 80)
        
    print("="*80)

if __name__ == "__main__":
    train_baselines(seqLength=5)
