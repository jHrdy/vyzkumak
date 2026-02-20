from pathlib import Path

# root (vyzkumak/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = PROJECT_ROOT / "plots"