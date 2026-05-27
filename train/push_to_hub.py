"""Push IRouterLM model to HuggingFace Hub from a training checkpoint."""
import os
import argparse
import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi
from train.hf_model import IRouterLMConfig, IRouterLMModel

REPO_ID = "ananoymous/IRouterLM"
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
STRATEGIES = ["MULTIMODAL_RERANK", "MULTIMODAL-SINGLE", "TEXT_RERANK", "TEXT-SINGLE"]
LORA_CONFIG = {"r": 16, "alpha": 32, "dropout": 0.05, 
               "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def load_checkpoint_and_merge(checkpoint_path: str):
    """Load checkpoint, apply LoRA, and merge weights."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load state dict
    safetensors = os.path.join(checkpoint_path, "model.safetensors")
    state_dict = load_file(safetensors) if os.path.exists(safetensors) else \
                 torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    # Get dimensions from classifier
    classifier_w = state_dict.get("classifier.weight")
    hidden_size = classifier_w.shape[1] if classifier_w is not None else 1024
    num_labels = classifier_w.shape[0] if classifier_w is not None else 4
    print(f"Detected: hidden_size={hidden_size}, num_labels={num_labels}")
    
    # Load base model + apply LoRA
    print("Loading base model and applying LoRA...")
    base = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=torch.float32)
    peft_config = LoraConfig(r=LORA_CONFIG["r"], lora_alpha=LORA_CONFIG["alpha"], 
                             target_modules=LORA_CONFIG["targets"], lora_dropout=LORA_CONFIG["dropout"],
                             bias="none", task_type=TaskType.FEATURE_EXTRACTION)
    peft_model = get_peft_model(base, peft_config)
    
    # Map and load LoRA weights
    peft_dict = {k.replace("base_model.base_model.", "base_model."): v 
                 for k, v in state_dict.items() if k not in ["classifier.weight", "classifier.bias"]}
    peft_model.load_state_dict(peft_dict, strict=False)
    
    # Merge LoRA
    print("Merging LoRA weights...")
    merged = peft_model.merge_and_unload()
    
    # Create IRouterLM model
    config = IRouterLMConfig(base_model_name=BASE_MODEL, hidden_size=hidden_size, 
                             num_labels=num_labels, strategy_names=STRATEGIES)
    model = IRouterLMModel(config)
    model.transformer = merged
    if classifier_w is not None:
        model.classifier.weight.data = classifier_w
    if state_dict.get("classifier.bias") is not None:
        model.classifier.bias.data = state_dict["classifier.bias"]
    
    print("Model assembled!")
    return model, tokenizer


def push_to_hub(model, tokenizer, repo_id: str, token: str = None):
    """Push model, tokenizer, and code to HuggingFace Hub."""
    print(f"\nPushing to: {repo_id}")
    IRouterLMConfig.register_for_auto_class()
    IRouterLMModel.register_for_auto_class("AutoModel")
    
    model.push_to_hub(repo_id, commit_message="Upload model", token=token)
    tokenizer.push_to_hub(repo_id, commit_message="Upload tokenizer", token=token)
    
    # Upload code files
    api = HfApi()
    hf_dir = os.path.join(os.path.dirname(__file__), "hf_model")
    for f in ["configuration_irouterlm.py", "modeling_irouterlm.py", "README.md"]:
        api.upload_file(path_or_fileobj=os.path.join(hf_dir, f), path_in_repo=f,
                        repo_id=repo_id, token=token, commit_message=f"Add {f}")
    
    print(f"\nDone! https://huggingface.co/{repo_id}")
    print(f'\nUsage:\n  model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)')


def main():
    parser = argparse.ArgumentParser(description="Push IRouterLM to HuggingFace Hub")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--repo-id", default=REPO_ID, help="HuggingFace repo ID")
    parser.add_argument("--token", help="HuggingFace API token")
    parser.add_argument("--dry-run", action="store_true", help="Test without pushing")
    args = parser.parse_args()
    
    print("=" * 60)
    print("IRouterLM - Push to HuggingFace Hub")
    print("=" * 60)
    
    model, tokenizer = load_checkpoint_and_merge(args.checkpoint)
    
    # Verify
    print("\nVerifying model...")
    inputs = tokenizer("What is the capital of France?", return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs)["logits"]
    probs = torch.softmax(logits, dim=-1)
    print(f"Predictions: {probs}")
    print(f"Strategy: {STRATEGIES[probs.argmax().item()]}")
    
    if not args.dry_run:
        push_to_hub(model, tokenizer, args.repo_id, args.token)


if __name__ == "__main__":
    main()
