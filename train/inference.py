import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from sklearn.metrics import accuracy_score
from datasets import Dataset
from tqdm import tqdm

from train.config import (
    MAX_LENGTH,
    LOG_FILE,
    PREDICTIONS_PATH,
    TEST_DATA_PATH,
    ARM_NAMES,
    EPS,
    device,
)

INFERENCE_BATCH_SIZE = 32

# HuggingFace Hub model ID
HF_MODEL_ID = "ananoymous/IRouterLM"

REWARD_COLS = [
    "MULTIMODAL_RERANK_reward",
    "MULTIMODAL-SINGLE_reward",
    "TEXT_RERANK_reward",
    "TEXT-SINGLE_reward",
]


def load_model(model_id: str = HF_MODEL_ID):
    """
    Load the IRouterLM model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Tuple of (model, tokenizer)
    """
    print("=" * 80)
    print("LOADING MODEL FROM HUGGINGFACE HUB")
    print("=" * 80)
    print(f"Model ID: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)

    model.eval()
    print(f"Model loaded from HuggingFace Hub: {model_id}")

    return model, tokenizer


def load_test_data(test_path: str = None):
    """Load test data from Excel file."""
    if test_path is None:
        test_path = TEST_DATA_PATH

    print("=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)

    test_df = pd.read_excel(test_path)
    print(f"Test samples: {len(test_df)}")

    return test_df


def create_test_dataset(test_df: pd.DataFrame, tokenizer):
    """Create tokenized dataset from test dataframe."""
    labels = [list(r) for r in test_df[REWARD_COLS].to_numpy()]

    ds = Dataset.from_dict({
        "text": test_df["query"].tolist(),
        "labels": labels
    })

    ds = ds.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        ),
        batched=True,
        desc="Tokenizing"
    )

    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def compute_metrics(eval_pred) -> dict:
    """
    Compute evaluation metrics based on NDCG scores.
    Compatible with HuggingFace Trainer's compute_metrics interface.

    Args:
        eval_pred: EvalPrediction object or tuple of (logits, true_ndcgs)

    Returns:
        Dictionary of metrics
    """
    logits, true_ndcgs = eval_pred
    pred_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    # Normalize NDCG scores for distribution comparison
    label_probs = np.array([
        r / (np.sum(r) + EPS) if np.sum(r) > 0 else np.ones(len(r)) / len(r)
        for r in true_ndcgs
    ])

    pred_labels = np.argmax(pred_probs, axis=1)
    true_labels = np.argmax(true_ndcgs, axis=1)
    accuracy = accuracy_score(true_labels, pred_labels)

    cosine = np.mean([
        np.dot(p, l) / (np.linalg.norm(p) * np.linalg.norm(l) + EPS)
        for p, l in zip(pred_probs, label_probs)
    ])

    predicted_ndcg = np.mean([true_ndcgs[i, pred_labels[i]] for i in range(len(pred_labels))])
    oracle_ndcg = np.mean(np.max(true_ndcgs, axis=1))
    regret = oracle_ndcg - predicted_ndcg
    normalized_regret = regret / (oracle_ndcg + EPS)

    return {
        "accuracy": accuracy,
        "cosine_similarity": cosine,
        "predicted_ndcg": predicted_ndcg,
        "oracle_ndcg": oracle_ndcg,
        "regret": regret,
        "normalized_regret": normalized_regret,
    }


@torch.no_grad()
def run_inference(model, tokenizer, test_df: pd.DataFrame, batch_size: int = INFERENCE_BATCH_SIZE) -> np.ndarray:
    """
    Run batched inference on test data.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_df: Test dataframe
        batch_size: Batch size for inference

    Returns:
        Prediction probabilities, shape (num_samples, num_classes)
    """
    print("=" * 80)
    print("RUNNING INFERENCE")
    print("=" * 80)

    test_dataset = create_test_dataset(test_df, tokenizer)

    # Remove labels column for DataLoader (not needed for inference)
    test_dataset = test_dataset.remove_columns(["labels"])

    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    all_logits = []

    for batch in tqdm(dataloader, desc="Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs["logits"].cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    pred_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    return pred_probs


def evaluate_model(model, tokenizer, test_df: pd.DataFrame, save_predictions: bool = True):
    """
    Evaluate model on test set and optionally save predictions.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_df: Test dataframe
        save_predictions: Whether to save predictions to file

    Returns:
        Dictionary of metrics
    """
    pred_probs = run_inference(model, tokenizer, test_df)
    true_ndcgs = test_df[REWARD_COLS].to_numpy()

    # Compute metrics
    logits = np.log(pred_probs + EPS)  # Convert back for compute_metrics
    metrics = compute_metrics((logits, true_ndcgs))

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Value':<15}")
    print("-" * 45)
    print(f"{'Accuracy (Pick Best)':<30} {metrics['accuracy']:.4f}")
    print(f"{'Predicted NDCG (Mean)':<30} {metrics['predicted_ndcg']:.4f}")
    print(f"{'Oracle NDCG (Mean)':<30} {metrics['oracle_ndcg']:.4f}")
    print(f"{'Regret (Absolute)':<30} {metrics['regret']:.4f}")
    print(f"{'Regret (Normalized %)':<30} {metrics['normalized_regret']*100:.2f}%")

    if save_predictions:
        # Add predictions to dataframe
        results_df = test_df.copy()
        for i, arm in enumerate(ARM_NAMES):
            results_df[f"pred_prob_{arm}"] = pred_probs[:, i]
            results_df[f"true_ndcg_{arm}"] = true_ndcgs[:, i]

        # Predicted vs True logic
        results_df["predicted_label"] = np.argmax(pred_probs, axis=1)
        results_df["predicted_strategy"] = [ARM_NAMES[i] for i in results_df["predicted_label"]]

        results_df["true_label"] = np.argmax(true_ndcgs, axis=1)
        results_df["true_strategy"] = [ARM_NAMES[i] for i in results_df["true_label"]]

        results_df["predicted_ndcg"] = [
            true_ndcgs[i, results_df["predicted_label"].iloc[i]]
            for i in range(len(results_df))
        ]

        results_df["oracle_ndcg"] = np.max(true_ndcgs, axis=1)
        results_df["regret"] = results_df["oracle_ndcg"] - results_df["predicted_ndcg"]

        # Save predictions
        os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
        results_df.to_excel(PREDICTIONS_PATH, index=False)
        print(f"\nPredictions saved to {PREDICTIONS_PATH}")

        # Save metrics log
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics log saved to {LOG_FILE}")

    return metrics


def main():
    """Main inference pipeline."""
    print("\n" + "=" * 80)
    print("RAG STRATEGY CLASSIFIER - INFERENCE")
    print("=" * 80)
    print(f"Model: {HF_MODEL_ID}")
    print(f"Device: {device}")
    print("=" * 80)

    # Load model from HuggingFace Hub
    model, tokenizer = load_model()

    # Load test data
    test_df = load_test_data()

    # Evaluate
    metrics = evaluate_model(model, tokenizer, test_df)

    return metrics


if __name__ == "__main__":
    main()
