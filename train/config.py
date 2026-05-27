import torch

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Configuration
RANDOM_SEED = 42  # Random seed for reproducibility
SAMPLE_RATE = 0.1  # Sample rate per dataset (10% of each dataset)

# Reward Configuration
LAMBDA = 0.1  # Trade-off between NDCG and latency (0.1 = 90% NDCG, 10% latency)
LAMBDA_TAG = f"l{int(LAMBDA * 100):02d}"

# Strategy Configuration
ARM_NAMES = [
    "MULTIMODAL_RERANK",
    "MULTIMODAL-SINGLE",
    "TEXT_RERANK",
    "TEXT-SINGLE"
]

# Output Paths
DATA_PATH = f"train/data/updated_dataset.xlsx"
TRAIN_DATA_PATH = f"train/data/train_dataset_{LAMBDA_TAG}.xlsx"
TEST_DATA_PATH = f"train/data/test_dataset_{LAMBDA_TAG}.xlsx"
VAL_DATA_PATH = f"train/data/val_dataset_{LAMBDA_TAG}.xlsx"
OUTPUT_DIR = f"./models/model_{LAMBDA_TAG}"
LOG_FILE = f"{OUTPUT_DIR}/training_metrics_{LAMBDA_TAG}.json"
PREDICTIONS_PATH = f"evaluation/predictions/predictions_{LAMBDA_TAG}.xlsx"

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MAX_LENGTH = 128
EPS = 1e-8

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 2
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
