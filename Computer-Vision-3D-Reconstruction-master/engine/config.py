import os
import json

# Get the directory of the current file (config.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the project root (assuming config.json is there)
project_root = os.path.join(current_dir, '..')
config_path = os.path.join(project_root, 'config.json')

with open(config_path, 'r') as file:
    config = json.load(file)