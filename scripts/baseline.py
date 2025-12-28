
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR, SAVED_MODELS_DIR, HEATMAP_DIR
from src.multiModel import loadAndPreprocessData, createMultimodalSequences, preloadHeatmaps, MultimodalDataset

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
    
    gamesData, _, featureCols, _ = loadAndPreprocessData(gamesPath, shotsPath, teamsPath, SEQ_LENGTH)
    heatmapCache = preloadHeatmaps(HEATMAP_DIR)
    
    trainData = gamesData[gamesData['SEASON_ID'].isin(TRAIN_SEASONS)]
    testData = gamesData[gamesData['SEASON_ID'].isin(TEST_SEASONS)]
    
    print("Generating Sequences...")
    pTrain, gTrain, xTrain, yTrain = createMultimodalSequences(trainData, None, SEQ_LENGTH, featureCols, TARGET_COLS)
    pTest, gTest, xTest, yTest = createMultimodalSequences(testData, None, SEQ_LENGTH, featureCols, TARGET_COLS)
    
    # Create DataLoaders for Batch Processing (Heatmap aggregation)
    trainDataset = MultimodalDataset(pTrain, gTrain, xTrain, heatmapCache, yTrain)
    testDataset = MultimodalDataset(pTest, gTest, xTest, heatmapCache, yTest)
    
    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    testLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Helper to extract features
    def extract(loader):
        vis_list, stat_list, y_list = [], [], []
        with torch.no_grad():
            for imgs, stats, targets in tqdm(loader, desc="Extracting"):
                # Vis: Average Pooling -> Flatten
                avgImg = torch.mean(imgs, dim=1) # (B, 2, 50, 50)
                flat_vis = avgImg.numpy().reshape(avgImg.shape[0], -1) 
                
                # Stats: Flatten
                flat_stat = stats.view(stats.shape[0], -1).numpy()
                
                vis_list.append(flat_vis)
                stat_list.append(flat_stat)
                y_list.append(targets.numpy())
        return np.vstack(vis_list), np.vstack(stat_list), np.vstack(y_list)

    print("Extracting Train Features...")
    X_train_vis_raw, X_train_stat, y_train = extract(trainLoader)
    print("Extracting Test Features...")
    X_test_vis_raw, X_test_stat, y_test = extract(testLoader)
    
    return X_train_vis_raw, X_train_stat, y_train, X_test_vis_raw, X_test_stat, y_test

def train_and_evaluate():
    # 1. Get Data
    X_train_vis_raw, X_train_stat, y_train, X_test_vis_raw, X_test_stat, y_test = get_sequences_and_features()

    # 2. PCA for Visual Features
    print("Fitting PCA on Visual Features...")
    scaler_vis = StandardScaler()
    X_train_vis_scaled = scaler_vis.fit_transform(X_train_vis_raw)
    X_test_vis_scaled = scaler_vis.transform(X_test_vis_raw)
    
    pca = PCA(n_components=50) # Reduce 5000 -> 50
    X_train_vis_pca = pca.fit_transform(X_train_vis_scaled)
    X_test_vis_pca = pca.transform(X_test_vis_scaled)
    print(f"PCA Explained Variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 3. Define Model Configs
    # Name -> {FeatureSet, ModelClass}
    # FeatureSet: 'NO_CNN' (Stats only), 'WITH_CNN' (Stats + PCA Heatmaps)
    
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
    report_lines.append("="*100)
    report_lines.append(f"{'Model':<30} | {'PTS MAE':<10} | {'AST MAE':<10} | {'REB MAE':<10} | {'AVG MAE':<10} | {'AVG STD':<10}")
    report_lines.append("-" * 100)

    for name, feat_type, model in configs:
        preds = np.zeros_like(y_test)
        
        if feat_type == "NAIVE":
            # Just predict mean of training set per target
            train_means = np.mean(y_train, axis=0)
            preds = np.tile(train_means, (len(y_test), 1))
            
        else:
            # Select Features
            X_tr = X_train_no_cnn if feat_type == "NO_CNN" else X_train_with_cnn
            X_te = X_test_no_cnn if feat_type == "NO_CNN" else X_test_with_cnn
            
            # Train & Predict per target (MultiOutput)
            # Scikit Learn LinearRegression supports multi-output natively
            # XGBoost needs MultiOutputRegressor OR loop.
            if isinstance(model, LinearRegression):
                model.fit(X_tr, y_train)
                preds = model.predict(X_te)
            else:
                # XGBoost Loop
                for i in range(3):
                    model.fit(X_tr, y_train[:, i])
                    preds[:, i] = model.predict(X_te)
        
        # Calculate Metrics
        try:
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
            sys.stdout.flush() # Force print
            report_lines.append(line)
            
        except Exception as e:
            err_msg = f"Error evaluating {name}: {str(e)}"
            print(err_msg)
            report_lines.append(err_msg)

    print("="*100)
    report_lines.append("="*100)
    
    # Save Report
    with open("baseline_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print("Report saved to baseline_report.txt")

if __name__ == "__main__":
    train_and_evaluate()
