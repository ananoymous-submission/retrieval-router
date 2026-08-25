import pandas as pd
from train.config import (
    DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    SAMPLE_RATE, RANDOM_SEED, LAMBDA, OBJECTIVE,
    NDCG_COLS, LATENCY_COLS, REWARD_COLS
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

    # Extract values (column order follows ARM_NAMES, so arm i == column i everywhere)
    ndcg_values = df[NDCG_COLS].values
    latency_values = df[LATENCY_COLS].values

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
    for i, col in enumerate(REWARD_COLS):
        df[col] = rewards[:, i]

    # Analysis columns
    arm_index = {col: i for i, col in enumerate(REWARD_COLS)}
    df["optimal_reward"] = df[REWARD_COLS].max(axis=1)
    df["optimal_arm"] = df[REWARD_COLS].idxmax(axis=1).map(arm_index)

    return df

def save_datasets(train_df, val_df, test_df, paths=None):
    train_path, val_path, test_path = paths or (TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH)
    print(f"\nSaving datasets...")
    train_df.to_excel(train_path, index=False)
    print(f"Saved training data to {train_path}")

    val_df.to_excel(val_path, index=False)
    print(f"Saved validation data to {val_path}")

    test_df.to_excel(test_path, index=False)
    print(f"Saved test data to {test_path}")


def main(objective=None, paths=None):
    """Split the data, then let the objective attach its own training target.

    The split is seeded and objective-independent, so every method -- ours and any baseline --
    trains and is evaluated on exactly the same queries.
    """
    from train.objectives import get_objective   # local: prepare_data <-> objectives cycle
    objective = objective or get_objective(OBJECTIVE)

    print(f"=== Dataset Preparation (objective: {objective.name}) ===\n")

    # Step 1: Split (shared, seeded, identical for every objective)
    train_df, val_df, test_df = load_and_split_data()

    # Step 2: Attach this objective's target
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\nBuilding targets for {name} set...")
        objective.add_targets(df)

    # Step 3: Save
    save_datasets(train_df, val_df, test_df, paths)

    print("\n=== Dataset Preparation Complete ===")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")


if __name__ == "__main__":
    main()