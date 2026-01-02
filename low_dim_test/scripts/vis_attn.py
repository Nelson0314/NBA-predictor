
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATA_DIR, SAVED_MODELS_DIR
# Import from seqModel
from src.seqModel import loadAndPreprocessData, createSequences, NbaTransformer

# ==========================================
# 0. Monkey Patch Transformer for Attention Weights
# ==========================================
from torch.nn.modules.transformer import TransformerEncoderLayer

# We patch the _sa_block to force need_weights=True.
# This works for both standard PyTorch Transformer and likely NbaTransformer since it uses TransformerEncoderLayer.
def _sa_block_patched(self, x, attn_mask, key_padding_mask, is_causal=False):
    x = self.self_attn(x, x, x,
                       attn_mask=attn_mask,
                       key_padding_mask=key_padding_mask,
                       is_causal=is_causal,
                       need_weights=True)[0]
    return self.dropout1(x)

TransformerEncoderLayer._sa_block = _sa_block_patched
print("Monkey patched TransformerEncoderLayer._sa_block to capture attention weights.")

# ==========================================
# 1. Simple Dataset (Stats Only)
# ==========================================
class SimpleSeqDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

    def __len__(self):
        return len(self.x)

def visualize_attention():
    print("========================================")
    print("Visualizing Attention Maps (Sequence Model)")
    print("========================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    # Identify Model in savedModels_conf/seq
    base_dir = SAVED_MODELS_DIR
    target_dir = os.path.join(base_dir, 'seq')
    
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    # Check if target_dir itself contains the model
    if os.path.exists(os.path.join(target_dir, 'config.json')):
        best_path = target_dir
        print(f"Found model directly in: {best_path}")
    else:
        # Find best model in subdirectories
        candidates = [os.path.join(target_dir, d) for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        if not candidates:
            print("No model folders found in seq directory and no direct model found.")
            return
            
        # Pick based on valid_loss if possible
        best_path = None
        best_score = float('inf')
        
        for p in candidates:
            pkl = os.path.join(p, 'config.json')
            if os.path.exists(pkl):
                try:
                    with open(pkl, 'r') as f: c = json.load(f)
                    s = c.get('valid_loss', float('inf'))
                    if s < best_score:
                        best_score = s
                        best_path = p
                except: pass
                
        if not best_path:
            best_path = candidates[0] # Fallback
        
    print(f"Loading Model from: {best_path}")
    
    with open(os.path.join(best_path, 'config.json'), 'r') as f:
        config = json.load(f)
        
    seqLength = config.get('seqLength', 7)
    print(f"Using seqLength: {seqLength}")
    
    seqLength = config.get('seqLength', 7)
    print(f"Using seqLength: {seqLength}")
    
    # Load Data (Strictly from games.csv as Test Set)
    datasetPath = config.get('datasetPath', os.path.join(DATA_DIR, 'games.csv'))
    if not os.path.exists(datasetPath):
        datasetPath = os.path.join(DATA_DIR, 'games.csv')
        
    print(f"Loading data from: {datasetPath}")
    gamesData, featureCols, targetCols = loadAndPreprocessData(datasetPath, seqLength)
    
    # Target Season: 22024 (Test Set)
    target_season = 22024
    print(f"Visualizing attention for Season ID: {target_season}")
    
    candidates_df = gamesData[gamesData['SEASON_ID'] == target_season]
    
    # Fallback if 22024 is empty (e.g., if games.csv stops at 2023)
    if candidates_df.empty:
        print(f"Warning: No data found for season {target_season}. Using metadata test seasons or all data.")
        test_seasons = config.get('testSeasons', [22024])
        candidates_df = gamesData[gamesData['SEASON_ID'].isin(test_seasons)]
        
    if len(candidates_df) < 100:
        candidates_df = gamesData
        
    # Filter Players with enough history (> seqLength+5)
    valid_players = candidates_df.groupby('Player_ID').filter(lambda x: len(x) > seqLength + 5)
    
    if valid_players.empty:
        valid_players = gamesData.groupby('Player_ID').filter(lambda x: len(x) > seqLength + 5)
        
    # Sample 5 Players -> NO, User wants full season.
    # pass_pids = valid_players['Player_ID'].unique()
    # sampled_pids = np.random.choice(pass_pids, min(len(pass_pids), 10), replace=False)
    
    # testData = valid_players[valid_players['Player_ID'].isin(sampled_pids)].copy()
    testData = valid_players.copy()
    print(f"Using full test set: {len(testData)} games from {testData['Player_ID'].nunique()} players.")
    
    # Create Sequences
    xTest, yTest, metaTest = createSequences(testData, seqLength, featureCols, targetCols)
    
    # Scale Features (Need to fit scaler on something... ideally training data)
    # But for visualization, scaling relative to this small batch is risky.
    # Let's fit scaler on the WHOLE filtered subset (or try to load scaler?)
    # Fitting on the whole valid_players subset is a decent approximation.
    scaler = None
    # Actually, seqModel.py fits scaler on Train Features.
    # We don't have Train data readily loaded here to fit perfectly.
    # We will fit on the entire `gamesData` (loaded above) to get global stats.
    # This is close enough.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(gamesData[featureCols].values)
    
    # Transform xTest
    N, S, F = xTest.shape
    xTestFlat = xTest.reshape(-1, F)
    xTestScaled = scaler.transform(xTestFlat).reshape(N, S, F)
    
    dataset = SimpleSeqDataset(xTestScaled, yTest) # y is irrelevant for attn visualization
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Init Model
    # NbaTransformer(inputDim, dModel, nHead, numLayers, outputDim, statEmbedDim, dropout)
    statEmbedDim = config.get('statEmbedDim', 128)
    
    model = NbaTransformer(
        inputDim=len(featureCols),
        dModel=config['dModel'],
        nHead=config['nHead'],
        numLayers=config['numLayers'],
        outputDim=len(targetCols),
        statEmbedDim=statEmbedDim,
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(best_path, 'model.ckpt'), map_location=device))
    model.eval()
    
    # Register Hook
    attention_weights = []
    def get_attention_hook(module, args, output):
        # TransformerEncoderLayer forward returns:
        # src (Tensor) – the output of the layer
        # But we monkey-patched _sa_block which is internal.
        # Wait, the HOOK is on what?
        # In previous script: last_layer.self_attn.register_forward_hook
        # nn.MultiheadAttention forward returns (attn_output, attn_output_weights)
        if isinstance(output, tuple) and len(output) == 2:
             attention_weights.append(output[1].detach().cpu().numpy())
    
    # Hook onto the last encoder layer's self_attn
    # model.transformerEncoder is nn.TransformerEncoder
    # .layers is a ModuleList of TransformerEncoderLayer
    last_layer = model.transformerEncoder.layers[-1]
    last_layer.self_attn.register_forward_hook(get_attention_hook)
    
    # Collect
    all_attentions = []
    print("Collecting attention maps...")
    
    for x in tqdm(loader):
        # x: (Batch, Seq, Feat)
        x = x[0] if isinstance(x, list) else x # Case if dataset returns tuple
        x = x.to(device)
        attention_weights.clear()
        
        with torch.no_grad():
            _ = model(x)
            
        if attention_weights:
            # attention_weights[0] is (Batch, Seq, Seq)
            all_attentions.append(attention_weights[0])
            
    if not all_attentions:
        print("Failed to capture attention weights.")
        return

    full_attn = np.concatenate(all_attentions, axis=0) # (Total, Seq, Seq)
    print(f"Aggregated Attention Shape: {full_attn.shape}")
    
    # 1. Global Average
    avg_map = np.mean(full_attn, axis=0)
    
    # 2. Final Step
    last_token_attn = full_attn[:, -1, :]
    avg_last_token = np.mean(last_token_attn, axis=0)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(avg_map, ax=axes[0], cmap='viridis', annot=True, fmt='.2f', square=True)
    axes[0].set_title("Global Average Attention Map (Seq Model)")
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")
    axes[0].invert_yaxis()
    
    x_indices = np.arange(len(avg_last_token))
    axes[1].bar(x_indices, avg_last_token, color='orange', edgecolor='black')
    axes[1].set_title("Average Attention of Final Prediction Step")
    axes[1].set_xlabel("Game History Index")
    axes[1].set_ylabel("Weight")
    axes[1].set_xticks(x_indices)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('attention_aggregated.png')
    print("Saved attention_aggregated.png")
    
    # Interpretation
    print("\n=== INTERPRETATION (Sequence Model) ===")
    print("1. Bar Chart (Right): Shows how much the LSTM/Transformer relies on past games.")
    print("   - This model only sees Stats (no Images).")
    print(f"   - Index {len(avg_last_token)-1} is the most recent game.")

if __name__ == "__main__":
    visualize_attention()
