import os

# Root of the project (one level up from src)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, 'savedModels_conf')

# Commonly used paths
GAMES_PATH = os.path.join(DATA_DIR, 'games.csv')
SHOTS_PATH = os.path.join(DATA_DIR, 'shots.csv')
TEAMS_PATH = os.path.join(DATA_DIR, 'teams.csv')
