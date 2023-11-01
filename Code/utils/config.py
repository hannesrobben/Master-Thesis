import os
import pathlib
root_dir = pathlib.Path(__file__).parent.resolve().parent

exp_name = 'Training'
base_dir = root_dir / 'TRAINING_DATA' / exp_name

SAVED_AGENTS_DIR = base_dir / 'saved_agents'
TENSORBOARD_LOG_DIR = base_dir / 'tensorboard_logs'
OPTUNA_DB = base_dir / 'optuna.db'
MLFLOW_RUNS_DIR = base_dir / 'mlflow_runs'
EVAL_IMG_DIR = base_dir / 'evaluation_img'
TRAIN_IMG_DIR = base_dir / 'train_img'

if not SAVED_AGENTS_DIR.exists():
    os.makedirs(SAVED_AGENTS_DIR)

if not TENSORBOARD_LOG_DIR.exists():
    os.makedirs(TENSORBOARD_LOG_DIR)

if not MLFLOW_RUNS_DIR.exists():
    os.makedirs(MLFLOW_RUNS_DIR)

if not EVAL_IMG_DIR.exists():
    os.makedirs(EVAL_IMG_DIR)

if not TRAIN_IMG_DIR.exists():
    os.makedirs(TRAIN_IMG_DIR)

if __name__ == "__main__" :
    print(base_dir)
    print(OPTUNA_DB)