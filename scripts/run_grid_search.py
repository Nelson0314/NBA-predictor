import torch
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import re
from importlib import reload
import train_with_conf 
from src.config import DATA_DIR 

# ==========================================
# 1. 配置實驗參數 (Grid Search)
# ==========================================
experiments = [
    {
        "name": "Exp7_UltraDeep",
        "params": ["--seqLength", "5", "--dModel", "256", "--nHead", "16", "--numLayers", "6", "--dropout", "0.4", "--weightDecay", "1e-3", "--nEpochs", "40"]
    },
    {
        "name": "Exp8_cnnBoost",
        "params": ["--seqLength", "5", "--dModel", "128", "--cnnEmbedDim", "128", "--numLayers", "4", "--dropout", "0.3", "--weightDecay", "1e-4", "--nEpochs", "40"]
    },
    {
        "name": "Exp9_LongDeep",
        "params": ["--seqLength", "10", "--dModel", "128", "--nHead", "8", "--numLayers", "4", "--dropout", "0.4", "--weightDecay", "1e-4", "--nEpochs", "35"]
    },
    {
        "name": "Exp10_HighReg",
        "params": ["--seqLength", "5", "--dModel", "64", "--dropout", "0.5", "--weightDecay", "5e-2", "--nEpochs", "50", "--learningRate", "5e-5"]
    }
]

# Get Absolute Path of CWD
CWD = os.getcwd()
COMMON_ARGS = [
    "--gamesPath", os.path.join(DATA_DIR, "games.csv"),
    "--shotsPath", os.path.join(DATA_DIR, "shots.csv"),
    "--teamsPath", os.path.join(DATA_DIR, "teams.csv"),
    "--heatmapDir", os.path.join(DATA_DIR, "heatmaps"), 
    "--saveDir", "savedModels_grid"
]

summary_file = "grid_search_summary_final.txt"
best_roi = -999.0
best_exp_name = ""
results = {}

with open(summary_file, "w") as f:
    f.write("=== FINAL GRID SEARCH REPORT ===\n\n")

print(f"Using Data Paths:\n{COMMON_ARGS[:4]}")

# ==========================================
# 2. 執行實驗
# ==========================================
for i, exp in enumerate(experiments):
    exp_name = exp["name"]
    print(f"\n\n{'='*50}")
    print(f"RUNNING EXPERIMENT {i+1}/{len(experiments)}: {exp_name}")
    print(f"Params: {exp['params']}")
    print(f"{'='*50}\n")
    
    save_path = os.path.join("..", "savedModels_grid", exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    run_args = ["train_with_conf.py"] + COMMON_ARGS + ["--saveDir", save_path] + exp["params"]
    sys.argv = run_args
    
    try:
        reload(train_with_conf)
        train_with_conf.train_and_simulate()
        
        report_path = os.path.join(save_path, "simulation_report.txt")
        roi = -999.0
        
        if os.path.exists(report_path):
            with open(report_path, "r") as r:
                content = r.read()
            match = re.search(r"ROI: ([\-\d\.]+)%", content)
            if match:
                roi = float(match.group(1))
            
            results[exp_name] = {'roi': roi, 'path': save_path}
            with open(summary_file, "a") as f:
                f.write(f"\n--- {exp_name} ---\n")
                f.write(f"ROI: {roi:.2f}%\n")
                f.write(f"Params: {exp['params']}\n")
                f.write("-" * 30 + "\n")
            print(f"✅ {exp_name} Finished. ROI: {roi:.2f}%")
            
            if roi > best_roi:
                best_roi = roi
                best_exp_name = exp_name
        else:
            print(f"⚠️ Report not found for {exp_name}")
            results[exp_name] = {'roi': -999.0, 'path': save_path}
            
    except Exception as e:
        print(f"❌ FAILED {exp_name}: {e}")
        import traceback
        traceback.print_exc()
        results[exp_name] = {'roi': -999.0, 'path': save_path}

# ==========================================
# 3. 比較與清理
# ==========================================
print(f"\n\n{'='*50}")
print("COMPARISON & CLEANUP")
print(f"{'='*50}")

if best_exp_name:
    print(f"🏆 WINNER: {best_exp_name} (ROI: {best_roi:.2f}%)")
    with open(summary_file, "a") as f:
        f.write(f"\n🏆 WINNER: {best_exp_name} (ROI: {best_roi:.2f}%)\n")

    for exp_name, data in results.items():
        if exp_name != best_exp_name:
            print(f"🗑️ Deleting inferior model: {exp_name} (ROI: {data['roi']:.2f}%)")
            try:
                if os.path.exists(data['path']):
                    shutil.rmtree(data['path'])
                    print("   Deleted.")
            except Exception as e:
                print(f"   Error deleting: {e}")
        else:
            print(f"✨ Keeping best model: {exp_name}")
else:
    print("No successful experiments found.")

print(f"\nGrid Search Complete.")
