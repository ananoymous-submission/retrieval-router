import pandas as pd
from train.config import (
    DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    SAMPLE_RATE, RANDOM_SEED, LAMBDA
)

def load_and_split_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH).dropna()
    if "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "query"}, inplace=True)

    print(f"Original dataset shape: {df.shape}")
    print(f"Datasets present: {df['DATASET'].unique().tolist()}")

    train_dfs = []
    val_dfs = []
    test_dfs = []

    for dataset_name in df['DATASET'].unique():
        dataset_df = df[df['DATASET'] == dataset_name]
        total_len = len(dataset_df)
        
        # Calculate sizes
        n_test = int(total_len * SAMPLE_RATE)
        n_val = int(total_len * SAMPLE_RATE)

        # Safety check for small datasets
        if n_test == 0 or n_val == 0:
            raise ValueError(
                f"Dataset '{dataset_name}' has {total_len} samples. "
                f"Sampling {SAMPLE_RATE*100}% results in 0 samples for test/val. "
                f"Increase SAMPLE_RATE or ensure dataset has enough samples."
            )
        
        if (n_test + n_val) >= total_len:
             raise ValueError(
                f"Dataset '{dataset_name}' is too small for current split rates. "
                f"Total: {total_len}, Test+Val required: {n_test + n_val}"
            )

        # 1. Sample Test set from full data
        test_df = dataset_df.sample(n=n_test, random_state=RANDOM_SEED)
        
        # 2. Drop Test samples to get remaining pool
        remaining_df = dataset_df.drop(test_df.index)
        
        # 3. Sample Validation set from remaining pool
        val_df = remaining_df.sample(n=n_val, random_state=RANDOM_SEED)
        
        # 4. Remaining samples go to Train
        train_df = remaining_df.drop(val_df.index)

        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)

        print(f"  {dataset_name}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

    train_data = pd.concat(train_dfs, ignore_index=True)
    val_data = pd.concat(val_dfs, ignore_index=True)
    test_data = pd.concat(test_dfs, ignore_index=True)

    print(f"\nTrain dataset shape: {train_data.shape}")
    print(f"Val dataset shape:   {val_data.shape}")
    print(f"Test dataset shape:  {test_data.shape}")

    return train_data, val_data, test_data


def calculate_rewards(df):
    """
    Calculate rewards using lambda-weighted formula:
    reward = (1 - lambda) * ndcg + lambda * (1 - normalized_latency)
    """
    
    # Define column names
    ndcg_cols = [
        "MULTIMODAL_RERANK_ndcg", "MULTIMODAL-SINGLE_ndcg",
        "TEXT_RERANK_ndcg", "TEXT-SINGLE_ndcg"
    ]

    latency_cols = [
        "MULTIMODAL_RERANK_latency", "MULTIMODAL-SINGLE_latency",
        "TEXT_RERANK_latency", "TEXT-SINGLE_latency"
    ]

    reward_cols = [
        "MULTIMODAL_RERANK_reward", "MULTIMODAL-SINGLE_reward",
        "TEXT_RERANK_reward", "TEXT-SINGLE_reward"
    ]

    # Extract values
    ndcg_values = df[ndcg_cols].values
    latency_values = df[latency_cols].values

    # Per-row normalize latency (This keeps latency relative to the query)
    latency_sum = latency_values.sum(axis=1, keepdims=True)
    normalized_latency = latency_values / (latency_sum + 1e-9)

    # Calculate raw rewards
    rewards = (1 - LAMBDA) * ndcg_values + LAMBDA * (1 - normalized_latency)

    # =========================================================================
    # FIX: REMOVED NORMALIZATION
    # We want the absolute values (e.g., 0.8 vs 0.1) so the Trainer knows
    # which queries are "high stakes" vs "noise".
    # =========================================================================

    # Assign RAW rewards back to DF
    df[reward_cols[0]] = rewards[:, 0]
    df[reward_cols[1]] = rewards[:, 1]
    df[reward_cols[2]] = rewards[:, 2]
    df[reward_cols[3]] = rewards[:, 3]

    # Analysis columns
    df["optimal_reward"] = df[reward_cols].max(axis=1)
    df["optimal_arm"] = df[reward_cols].idxmax(axis=1).map({
        "MULTIMODAL_RERANK_reward": 0,
        "MULTIMODAL-SINGLE_reward": 1,
        "TEXT_RERANK_reward": 2,
        "TEXT-SINGLE_reward": 3
    })
    
    return df

def save_datasets(train_df, val_df, test_df):
    print(f"\nSaving datasets...")
    train_df.to_excel(TRAIN_DATA_PATH, index=False)
    print(f"Saved training data to {TRAIN_DATA_PATH}")
    
    val_df.to_excel(VAL_DATA_PATH, index=False)
    print(f"Saved validation data to {VAL_DATA_PATH}")

    test_df.to_excel(TEST_DATA_PATH, index=False)
    print(f"Saved test data to {TEST_DATA_PATH}")


def main():
    print("=== Dataset Preparation for Neural Multi-Armed Bandit ===\n")

    # Step 1: Split
    train_df, val_df, test_df = load_and_split_data()
    
    # Step 2: Calculate Rewards
    print("\nCalculating rewards for Train set...")
    train_df = calculate_rewards(train_df)
    
    print("Calculating rewards for Val set...")
    val_df = calculate_rewards(val_df)
    
    print("Calculating rewards for Test set...")
    test_df = calculate_rewards(test_df)

    # Step 3: Save
    save_datasets(train_df, val_df, test_df)

    print("\n=== Dataset Preparation Complete ===")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")


if __name__ == "__main__":
    main()