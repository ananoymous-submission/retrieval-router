import os
import torch

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Configuration
RANDOM_SEED = 42  # Random seed for reproducibility
SAMPLE_RATE = 0.1  # Sample rate per dataset (10% of each dataset)

# Reward Configuration
LAMBDA = float(os.getenv("LAMBDA", "0.1"))  # NDCG/latency trade-off; override via env for a sweep
# RUN_SUFFIX keeps a run's artifacts/checkpoints distinct (e.g. same lambda, different data).
LAMBDA_TAG = f"l{int(LAMBDA * 100):02d}{os.getenv('RUN_SUFFIX', '')}"

# Target temperature for softmax(reward / TEMPERATURE); sharpens small reward gaps.
TEMPERATURE = 0.1

# Pipeline Configuration — defined once in the top-level arms module and re-exported here,
# so train/ and evaluation/ can never disagree about the arm set or its order.
from arms import ARM_NAMES, NDCG_COLS, LATENCY_COLS, REWARD_COLS  # noqa: F401

# Which training signal to run. One file per approach in train/objectives/:
#   "retrievalrouter" -> RetrievalRouter: full reward vector + KL
#   "baseline"        -> Arabzadeh et al. CIKM 2021 adaptation: hard label + cross-entropy
# Everything else about the two runs is shared, so this switch is the whole difference.
OBJECTIVE = os.getenv("OBJECTIVE", "retrievalrouter")

# Names every artifact of a run using the public method names.
DEFAULT_RUN_TAG = "baseline" if OBJECTIVE == "baseline" else f"retrievalrouter_{LAMBDA_TAG}"
RUN_TAG = os.getenv("RUN_TAG", DEFAULT_RUN_TAG)

# Output Paths
DATA_PATH = os.getenv("TRAIN_DATA_FILE", "train/data/updated_dataset.xlsx")
TRAIN_DATA_PATH = f"train/data/train_dataset_{RUN_TAG}.xlsx"
TEST_DATA_PATH = f"train/data/test_dataset_{RUN_TAG}.xlsx"
VAL_DATA_PATH = f"train/data/val_dataset_{RUN_TAG}.xlsx"
OUTPUT_DIR = f"./models/model_{RUN_TAG}"
LOG_FILE = f"{OUTPUT_DIR}/training_metrics_{RUN_TAG}.json"
PREDICTIONS_PATH = f"evaluation/predictions/{RUN_TAG}.xlsx"

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MAX_LENGTH = 128
EPS = 1e-8

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = int(os.getenv("EPOCHS", "2"))
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Precision
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float32
