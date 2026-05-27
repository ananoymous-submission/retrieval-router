import os
from datasets import load_dataset, concatenate_datasets

def load_dataset_for_benchmark(dataset_name: str):
    dataset = load_dataset(dataset_name, token=os.getenv("HF_TOKEN"))
    
    all_splits = []
    for key in dataset.keys():
        all_splits.append(dataset[key])
    
    dataset_across_splits = concatenate_datasets(all_splits)    
    return dataset_across_splits