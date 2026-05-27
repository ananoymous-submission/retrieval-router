import os
import json
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def load_queries(task, pipeline, query_column):
    with open(f"qrels/{task}/{pipeline}/per_query/{query_column}.json_metrics_per_query_top5.json", "r") as f:
        data = json.load(f)
    return data

def load_latencies(task, pipeline, query_column):
    if pipeline != "BM25":
        with open(f"qrels/{task}/{pipeline}/{query_column}.json", "r") as f:
            data = json.load(f)
        data = {k: v["search_latency"] for k, v in data.items()}
    else:
        with open(f"qrels/{task}/{pipeline}/{query_column}.json", "r") as f:
            data = json.load(f)
        data = {k: 0 for k in data.keys()}
    return data

tasks = ["finreport","finslides","finqa", "convfinqa", "vqaonbd","tatdqa","arxivqa","dude","mp-docvqa","sciqag","wiki-ss"]
pipelines = ["MULTIMODAL-MULTI", "MULTIMODAL-SINGLE", "TEXT-MULTI", "TEXT-SINGLE"]

def process_task_pipeline(task, pipeline, query_column):
    """Process a single task-pipeline-query combination and return results."""
    try:
        metrics_data = load_queries(task, pipeline, query_column)
        latency_data = load_latencies(task, pipeline, query_column)

        results = []
        for query, metrics in metrics_data.items():
            result = {
                "query": query,
                pipeline + "_ndcg": metrics["ndcg"],
                pipeline + "_mrr": metrics["mrr"],
                pipeline + "_precision": metrics["precision"],
                pipeline + "_recall": metrics["recall"],
                pipeline + "_latency": latency_data[query],
                "DATASET": task
            }
            results.append(result)
        return results
    except Exception as e:
        if query_column == "query":
            print(f"Error loading data for {task} {pipeline}: {e}")
        return []

def main():
    columns = []

    for pipeline in pipelines:
        columns.append(pipeline + "_ndcg")
        columns.append(pipeline + "_mrr")
        columns.append(pipeline + "_precision")
        columns.append(pipeline + "_recall")
        columns.append(pipeline + "_latency")

    # Generate all combinations to process
    combinations = [
        (task, pipeline, query_column)
        for task in tasks
        for pipeline in pipelines
        for query_column in ["query"]
    ]

    # Process all combinations in parallel with 16 jobs
    all_results = Parallel(n_jobs=16)(
        delayed(process_task_pipeline)(task, pipeline, query_column)
        for task, pipeline, query_column in tqdm(combinations, desc="Processing datasets")
    )

    # Flatten results and create DataFrame
    flattened_results = [item for sublist in all_results for item in sublist]

    # Group results by query to combine metrics from different pipelines
    query_data = {}
    for result in flattened_results:
        query = result.pop("query")
        if query not in query_data:
            query_data[query] = {}
        query_data[query].update(result)

    # Create DataFrame from aggregated data
    performance_df = pd.DataFrame.from_dict(query_data, orient='index')
    performance_df = performance_df[columns + ["DATASET"]]

    performance_df.to_excel("train/dataset.xlsx")

if __name__ == "__main__":
    main()