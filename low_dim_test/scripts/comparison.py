import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Insert test/src into path
sys.path.insert(0, os.path.abspath('test/src'))

from multiModel import loadAndPreprocessData, createMultimodalSequences, preloadHeatmaps, MultimodalDataset, NbaMultimodal
from seqModel import NbaTransformer
from config import GAMES_PATH, SHOTS_PATH, TEAMS_PATH, HEATMAP_DIR

def evaluate_dl_model(name, model, testLoader, device, scalerY):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for items in testLoader:
            if len(items) == 3:
                imgs, stats, targets = items
                imgs, stats = imgs.to(device), stats.to(device)
                out = model(imgs, stats)
            else:
                stats, targets = items
                stats = stats.to(device)
                out = model(stats)
                
            if out.dim() == 2 and out.size(1) == 9: 
                out = out.view(out.size(0), 3, 3)
            
            if out.dim() == 3 and out.size(2) == 3: 
                p50 = out[:, :, 1].cpu().numpy()
            else: 
                p50 = out.cpu().numpy()
                
            all_preds.append(p50)
            all_targets.append(targets[:, :3].numpy() if targets.dim() > 1 else targets.numpy())
            
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    preds_orig = scalerY.inverse_transform(preds)
    
    maes = np.mean(np.abs(preds_orig - targets), axis=0)
    avg_mae = np.mean(maes)
    return f'{name:<30} | {maes[0]:<10.4f} | {maes[1]:<10.4f} | {maes[2]:<10.4f} | {avg_mae:<10.4f} | {np.std(maes):<10.4f}', maes

def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    trainSeasons = [22016, 22017, 22018, 22019, 22021, 22022]
    testSeasons = [22024]
    SEQ_LENGTH = 7
    predictCols = ['PTS', 'AST', 'REB']
    
    print('Loading data...')
    gamesData, shotsGrouped, featureCols, _ = loadAndPreprocessData(GAMES_PATH, SHOTS_PATH, TEAMS_PATH, seqLength=SEQ_LENGTH)
    trData = gamesData[gamesData['SEASON_ID'].isin(trainSeasons)]
    teData = gamesData[gamesData['SEASON_ID'].isin(testSeasons)]
    
    print('Creating sequences...')
    xP_tr, xG_tr, xS_tr, y_tr = createMultimodalSequences(trData, shotsGrouped, SEQ_LENGTH, featureCols, predictCols)
    xP_te, xG_te, xS_te, y_te = createMultimodalSequences(teData, shotsGrouped, SEQ_LENGTH, featureCols, predictCols)
    
    y_tr = np.array(y_tr, dtype=np.float32)
    y_te = np.array(y_te, dtype=np.float32)
    
    print('Scaling...')
    num_feats = len(featureCols)
    scalerX = StandardScaler().fit(xS_tr.reshape(-1, num_feats))
    xS_te_s = scalerX.transform(xS_te.reshape(-1, num_feats)).reshape(xS_te.shape)
    
    # [Target Scaler Mismatch Fix]
    # MultiModel: MinMaxScaler(0,1)
    scalerY_mm = MinMaxScaler(feature_range=(0, 1)).fit(y_tr[:, :3])
    # SeqModel: StandardScaler (Z-score)
    scalerY_std = StandardScaler().fit(y_tr[:, :3])
    
    print('Preloading heatmaps...')
    heatmapCache = preloadHeatmaps(HEATMAP_DIR)
    
    print('Evaluating Models...')
    report = ['='*100, f'{"Model":<30} | {"PTS MAE":<10} | {"AST MAE":<10} | {"REB MAE":<10} | {"AVG MAE":<10} | {"AVG STD":<10}', '-'*100]
    
    models = [
        ('Transformer Seq (Full)', 'test/savedModels_7dim/seq'), 
        ('Transformer Multi (Full)', 'test/savedModels_7dim/multi')
    ]
    
    for name, path in models:
        conf_path = os.path.join(path, 'config.json')
        cp_path = os.path.join(path, 'model.ckpt')
        
        if not os.path.exists(conf_path) or not os.path.exists(cp_path):
            print(f"Skipping {name}, missing files in {path}")
            continue
            
        with open(conf_path, 'r') as f:
            conf = json.load(f)
        
        if 'multi' in name.lower():
            model = NbaMultimodal(
                numStatFeatures=num_feats,
                seqLength=SEQ_LENGTH,
                outputDim=9,
                dModel=conf.get('dModel', 64),
                nHead=conf.get('nHead', 8),
                numLayers=conf.get('numLayers', 3),
                cnnEmbedDim=conf.get('cnnEmbedDim', 64)
            )
            testDs = MultimodalDataset(xP_te, xG_te, xS_te_s, heatmapCache, y=y_te)
            target_scaler = scalerY_mm
        else:
            model = NbaTransformer(
                inputDim=num_feats,
                outputDim=3,
                dModel=conf.get('dModel', 64),
                nHead=conf.get('nHead', 8),
                numLayers=conf.get('numLayers', 3)
            )
            # Standard Seq model uses xS_te_s and y_te Scaled with StandardScaler
            # But the evaluate function expects RAW y in testLoader if it does inverse_transform
            # Wait, our simple evaluate_dl_model pulls y from testLoader.
            # MultimodalDataset returns Raw y. TensorDataset should too.
            testDs = TensorDataset(torch.FloatTensor(xS_te_s), torch.FloatTensor(y_te))
            target_scaler = scalerY_std
            
        model.load_state_dict(torch.load(cp_path, map_location=device))
        model.to(device)
        
        testLoader = DataLoader(testDs, batch_size=64, shuffle=False)
        line, _ = evaluate_dl_model(name, model, testLoader, device, target_scaler)
        print(line)
        report.append(line)
        
    report.append('='*100)
    
    with open('test/baseline_report_all.txt', 'w') as f:
        f.write('\n'.join(report))
    print('\nReport saved to test/baseline_report_all.txt')

if __name__ == '__main__':
    train_and_evaluate()
