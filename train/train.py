import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from train.config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    VAL_DATA_PATH,
    ARM_NAMES,
    MODEL_NAME,
    MAX_LENGTH,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    OUTPUT_DIR,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    USE_BF16,
    DTYPE,
    device,
)
from train.model import QwenForSoftClassification
from train.inference import evaluate_model, compute_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

REWARD_COLS = [
    "MULTIMODAL_RERANK_reward",
    "MULTIMODAL-SINGLE_reward",
    "TEXT_RERANK_reward",
    "TEXT-SINGLE_reward",
]

def load_data():
    """Load training, validation, and test data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    train_df = pd.read_excel(TRAIN_DATA_PATH)
    val_df = pd.read_excel(VAL_DATA_PATH)
    test_df = pd.read_excel(TEST_DATA_PATH)

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    print(f"Features: query")
    print(f"Targets: {ARM_NAMES}")
    return train_df, val_df, test_df


def create_datasets(train_df, val_df, test_df, tokenizer):
    """Create tokenized datasets with soft labels (reward scores)."""
    print("\n" + "=" * 80)
    print("PREPARING DATASETS")
    print("=" * 80)

    def df_to_dataset(df):
        labels = [list(r) for r in df[REWARD_COLS].to_numpy()]

        ds = Dataset.from_dict({
            "text": df["query"].tolist(),
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

    train_dataset = df_to_dataset(train_df)
    val_dataset = df_to_dataset(val_df)
    test_dataset = df_to_dataset(test_df)

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset:   {len(val_dataset)} samples")
    print(f"Test dataset:  {len(test_dataset)} samples")
    print(f"Max sequence length: {MAX_LENGTH}")
    return train_dataset, val_dataset, test_dataset


class SoftLabelTrainer(Trainer):
    """Custom trainer for soft label learning."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        device = next(model.parameters()).device

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32, device=device)
        else:
            labels = labels.to(dtype=torch.float32, device=device)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )

        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss


def main():
    """Main training pipeline."""
    print("\n" + "=" * 80)
    print("RAG STRATEGY CLASSIFIER - QWEN3-0.6B FINE-TUNING")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Precision: {DTYPE}")
    print(f"LoRA Rank: {LORA_R}, Alpha: {LORA_ALPHA}")
    print("=" * 80)

    # 1. Load data
    train_df, val_df, test_df = load_data()

    # 2. Load tokenizer
    print("\n" + "=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(train_df, val_df, test_df, tokenizer)

    # 4. Load base model
    print("\n" + "=" * 80)
    print("LOADING BASE MODEL")
    print("=" * 80)
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=DTYPE,
    ).to(device)

    # 5. Apply LoRA
    print("\n" + "=" * 80)
    print("APPLYING LoRA")
    print("=" * 80)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()

    # 6. Create classification model
    print("\n" + "=" * 80)
    print("CREATING CLASSIFICATION MODEL")
    print("=" * 80)
    model = QwenForSoftClassification(
        base_model=base_model,
        num_labels=len(ARM_NAMES),
        hidden_size=base_model.config.hidden_size,
        dropout=0.1
    ).to(device)

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Training
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,

        # Optimization
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",

        # Precision
        bf16=USE_BF16,
        fp16=False if USE_BF16 else True,

        # Evaluation (ON VALIDATION SET)
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="predicted_ndcg",
        greater_is_better=True,

        # Logging
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        logging_first_step=True,
        report_to=["tensorboard"],

        # Misc
        save_total_limit=2,
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # 8. Create trainer
    trainer = SoftLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # 9. Train
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    trainer.train()

    # 10. Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.base_model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    classifier_path = os.path.join(adapter_path, "classifier.pt")
    torch.save(model.classifier.state_dict(), classifier_path)

    print(f"LoRA adapter saved to {adapter_path}")
    print(f"Classifier weights saved to {classifier_path}")


if __name__ == "__main__":
    main()
