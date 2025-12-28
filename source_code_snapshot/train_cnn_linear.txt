
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR, SAVED_MODELS_DIR, HEATMAP_DIR
from src.multiModel import loadAndPreprocessData, createMultimodalSequences, preloadHeatmaps, MultimodalDataset, CnnEncoder

def extract_features_and_targets(loader, device, cnn_encoder=None):
    """
    Extracts features: [CNN_Features (if encoder) OR Flattened_Img, Flattened_Stats]
    """
    all_feats = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, stats, targets in tqdm(loader, desc="Extracting Features"):
            # imgs: (B, Seq, 2, 50, 50)
            # stats: (B, Seq, Feats)
            
            # 1. Average Pooling on Images
            avgImg = torch.mean(imgs, dim=1) # (B, 2, 50, 50)
            
            if cnn_encoder:
                # Use Randomly Initialized CNN as feature extractor (Random Projection)
                # Or if loaded, use learned features.
                # Assuming random for now as "Baseline" implies no pre-training usually unless specified.
                avgImg = avgImg.to(device)
                visual_feats = cnn_encoder(avgImg).cpu().numpy() # (B, EmbedDim)
            else:
                # Just flatten the image
                # (B, 2*50*50) = (B, 5000)
                visual_feats = avgImg.numpy().reshape(avgImg.shape[0], -1)
            
            # 2. Flatten Stats
            # (B, Seq*Feats)
            stat_feats = stats.view(stats.shape[0], -1).numpy()
            
            # 3. Concatenate
            # [Img, Stats]
            concat = np.hstack([visual_feats, stat_feats])
            
            all_feats.append(concat)
            all_targets.append(targets.numpy())
            
    return np.vstack(all_feats), np.vstack(all_targets)

def train_cnn_linear_sklearn():
    print("========================================")
    print("Running Experiment: Average Heatmap + Linear Regression (Sklearn)")
    print("========================================")
    
    # Config
    seqLength = 5
    batchSize = 64
    device = torch.device('cpu') # Extraction on CPU is fine, or GPU if CNN used
    
    # Files
    gamesPath = os.path.join(DATA_DIR, 'games.csv')
    shotsPath = os.path.join(DATA_DIR, 'shots.csv')
    teamsPath = os.path.join(DATA_DIR, 'teams.csv')
    
    # 1. Load Data
    gamesData, _, featureCols, targetCols = loadAndPreprocessData(gamesPath, shotsPath, teamsPath, seqLength)
    
    # 2. Heatmaps
    heatmapCache = preloadHeatmaps(HEATMAP_DIR)
    
    # 3. Split
    trainSeasons = [22016, 22017, 22018, 22019, 22020, 22021, 22022]
    testSeasons = [22024]
    
    trainData = gamesData[gamesData['SEASON_ID'].isin(trainSeasons)]
    testData = gamesData[gamesData['SEASON_ID'].isin(testSeasons)]
    
    print("Generating Train Sequences...")
    pTrain, gTrain, xTrain, yTrain = createMultimodalSequences(trainData, None, seqLength, featureCols, targetCols)
    print("Generating Test Sequences...")
    pTest, gTest, xTest, yTest = createMultimodalSequences(testData, None, seqLength, featureCols, targetCols)
    
    # Dataset
    trainDataset = MultimodalDataset(pTrain, gTrain, xTrain, heatmapCache, yTrain)
    testDataset = MultimodalDataset(pTest, gTest, xTest, heatmapCache, yTest)
    
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=False, num_workers=0)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False, num_workers=0)
    
    # 4. Feature Extraction
    print("\nExtracting Features...")
    # Modify extraction to return separated visual and stat features
    def get_separated_features(loader, device):
        vis_list, stat_list, y_list = [], [], []
        with torch.no_grad():
            for imgs, stats, targets in loader:
                avgImg = torch.mean(imgs, dim=1) # (B, 2, 50, 50)
                flat_vis = avgImg.numpy().reshape(avgImg.shape[0], -1)
                flat_stat = stats.view(stats.shape[0], -1).numpy()
                
                vis_list.append(flat_vis)
                stat_list.append(flat_stat)
                y_list.append(targets.numpy())
        return np.vstack(vis_list), np.vstack(stat_list), np.vstack(y_list)

    X_train_vis, X_train_stat, y_train = get_separated_features(trainLoader, device)
    X_test_vis, X_test_stat, y_test = get_separated_features(testLoader, device)

    # 5. PCA on Visual Features
    print("Fitting PCA on Visual Features (Heatmaps)...")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Scale first (important for PCA)
    scaler_vis = StandardScaler()
    X_train_vis_scaled = scaler_vis.fit_transform(X_train_vis)
    X_test_vis_scaled = scaler_vis.transform(X_test_vis)
    
    # PCA
    n_components = 50
    pca = PCA(n_components=n_components)
    X_train_vis_pca = pca.fit_transform(X_train_vis_scaled)
    X_test_vis_pca = pca.transform(X_test_vis_scaled)
    
    print(f"Explained Variance Ratio (Top {n_components}): {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Concatenate
    X_train = np.hstack([X_train_vis_pca, X_train_stat])
    X_test = np.hstack([X_test_vis_pca, X_test_stat])
    
    print(f"Final Train Feature Shape: {X_train.shape}")
    
    # 6. Sklearn Linear Regression
    print("\nTraining Sklearn Linear Regression...")
    lr = LinearRegression() # Defaults
    lr.fit(X_train, y_train)
    
    print("Predicting...")
    preds = lr.predict(X_test)
    
    # 6. Report
    print("\n" + "="*80)
    print("CNN (Flattened) + LINEAR REGRESSION REPORT")
    print("="*80)
    print(f"{'Metric':<10} | {'PTS':<10} | {'AST':<10} | {'REB':<10} | {'OVERALL':<10}")
    print("-" * 80)
    
    maes = []
    mses = []
    
    for i, tgt in enumerate(targetCols):
        mae = mean_absolute_error(y_test[:, i], preds[:, i])
        mse = mean_squared_error(y_test[:, i], preds[:, i])
        maes.append(mae)
        mses.append(mse)
        
    avg_mae = np.mean(maes)
    avg_mse = np.mean(mses)
    
    print(f"{'MAE':<10} | {maes[0]:<10.4f} | {maes[1]:<10.4f} | {maes[2]:<10.4f} | {avg_mae:<10.4f}")
    print(f"{'MSE':<10} | {mses[0]:<10.4f} | {mses[1]:<10.4f} | {mses[2]:<10.4f} | {avg_mse:<10.4f}")
    print("="*80)

if __name__ == "__main__":
    train_cnn_linear_sklearn()
