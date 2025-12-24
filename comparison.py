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
from seqModel import loadAndPreprocessData, createSequences

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

    # 1. Load Data (Reusing seqModel logic for consistency)
    print("Loading data for baselines...")
    # Use datasetPath if available, else gamesPath
    dataPath = config.get('datasetPath', config.get('gamesPath'))
    
    # We use the same seqLength as the config to ensure fair comparison of "input information"
    seqLength = config['seqLength']
    
    gamesData, featureCols, targetCols = loadAndPreprocessData(dataPath, seqLength)

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
            'mse': mse_per_target
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
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(xTrainFlat, yTrain)
    yPredLR = lr.predict(xTestFlat)
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
    print("\nTraining XGBoost...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1, random_state=config['seed'])
    xgb.fit(xTrainFlat, yTrain)
    yPredXGB = xgb.predict(xTestFlat)
    # XGBoost supports multi-output regression automatically via sklearn wrapper
    calculate_metrics(yTest, yPredXGB, 'XGBoost')

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

    # 7. Generate Report & Plots
    generate_report(results, config)


def generate_report(results, config):
    print("\nGenerating Report and Charts...")
    saveDir = config['saveDir']
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    # --- Bar Chart ---
    targets = ['PTS', 'AST', 'REB']
    models = list(results.keys())
    
    # Sort models: Baselines first, then DL
    # Simple heuristic/sort
    models.sort() 

    # Prepare data for plotting
    # data structure: { 'PTS': [score_model1, score_model2...], ... }
    plot_data = {t: [] for t in targets}
    
    for model in models:
        for t in targets:
            val = results[model]['rmse'].get(t, 0)
            plot_data[t].append(val)

    x = np.arange(len(models))  # label locations
    width = 0.25  # the width of the bars
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
        
        f.write("\n## 2. Visualization\n")
        f.write("![Comparison Chart](comparison_chart.png)\n")
        
        f.write("\n## 3. Configuration & Parameters\n")
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
