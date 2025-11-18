import os

# Get the directory containing this file (project root)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Config directory is at the root level
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "config.yaml")

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
