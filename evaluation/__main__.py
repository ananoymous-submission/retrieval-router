import os
import json
import logging
import asyncio
import concurrent.futures
from functools import lru_cache

import numpy as np
import pandas as pd
import pytrec_eval
from datasets import load_dataset
from evaluation.significance_tests import compare_models

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_dataset_cache(task: str):
    """
    Load dataset once and cache it for reuse.
    Subsequent calls with the same `task` will return the in-memory copy.
    """
    try:
        ds = load_dataset(f"ibm-research/REAL-MM-RAG_{task}", split="test")
    except Exception as e:
        ds = load_dataset(f"ananoymous/{task}", split="train")

    return ds.select_columns([
        "query",
        "image_filename"
    ])


def get_expected_qrels(
    task: str,
    query_column: str,
    doc_column: str = "image_filename",
    dataset_cache=None,
    top_k: int = 5
) -> dict:
    """
    Load relevance judgments for a given query_column from the HF dataset,
    save them to disk, and return as a dict[q][doc] = 1.
    """
    # Paths for caching on disk - updated to new structure
    out_dir = os.path.join(parent_dir, f"qrels/{task.lower()}")
    os.makedirs(out_dir, exist_ok=True)
    qrels_path = os.path.join(out_dir, f"expected_qrels_{query_column}_top{top_k}.json")
    
    # If we've already written this file, load and return it
    if os.path.exists(qrels_path):
        with open(qrels_path, "r") as f:
            qrels = json.load(f)
        logger.info(
            f"Loaded cached expected_qrels for '{query_column}' "
            f"({len(qrels)} queries) from {out_dir}"
        )
        return qrels
    
    # Otherwise build it from the in-memory dataset
    ds = dataset_cache  # no fallback to load_dataset, rely on caller
    qrels = {}
    for row in ds:
        q = row.get(query_column)
        d = row.get(doc_column)
        if q is None or d is None:
            continue
        qrels.setdefault(q, {})[d] = 1

    with open(qrels_path, "w") as f:
        json.dump(qrels, f, indent=4)
    logger.info(
        f"Saved expected_qrels for '{query_column}' "
        f"({len(qrels)} queries) to {out_dir}"
    )
    return qrels


def evaluate_file(task, directory, fname, dataset_cache, top_k):
    """Evaluate a single run file"""
    query_column = os.path.splitext(fname)[0]
    expected_qrels = get_expected_qrels(
        task,
        query_column,
        dataset_cache=dataset_cache,
        top_k=top_k
    )

    # Updated path structure - no more retrieval subfolder
    base = os.path.join(parent_dir, f"qrels/{task.lower()}/{directory}")
    run_path = os.path.join(base, fname)
    with open(run_path) as f:
        run_data = json.load(f)

    # Extract actual qrels from nested structure - handle new format with metadata
    run_qrels = {}
    for query, data in run_data.items():
        if isinstance(data, dict) and 'results' in data:
            # New format with metadata - extract the nested results
            run_qrels[query] = data['results']
        else:
            # Old format - use data directly
            run_qrels[query] = data

    # keep only queries with gold judgments
    run_qrels = {q: docs for q, docs in run_qrels.items() if q in expected_qrels}
    # truncate to top-k predictions
    truncated = {
        q: dict(sorted(docs.items(), key=lambda x: x[1], reverse=True)[:top_k])
        for q, docs in run_qrels.items()
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        expected_qrels,
        {'ndcg', 'recip_rank', f'P_{top_k}', f'recall_{top_k}'}
    )
    eval_res = evaluator.evaluate(truncated)

    # compute mean & std for each metric
    nd = np.array([m['ndcg'] for m in eval_res.values()])
    rr = np.array([m['recip_rank'] for m in eval_res.values()])
    pk = np.array([m[f'P_{top_k}'] for m in eval_res.values()])
    rk = np.array([m[f'recall_{top_k}'] for m in eval_res.values()])

    logger.info(
        f"Evaluated {directory}{fname}: "
        f"nDCG@{top_k}={nd.mean():.4f} ± {nd.std():.4f}"
    )
    
    return {
        'fname': fname,
        'ndcg': (nd.mean(), nd.std()),
        'mrr': (rr.mean(), rr.std()),
        'prec': (pk.mean(), pk.std()),
        'rec': (rk.mean(), rk.std()),
        'eval_res': eval_res
    }


def evaluate_pipeline(task: str, directory: str, dataset_cache=None, top_k: int = 5):
    """
    Evaluate all JSON run-files in `qrels/{task}/{directory}`,
    producing per-file metrics and raw per-query evals.
    Returns:
      - ndcg_scores, mrr_scores, prec_scores, rec_scores: dict[file] = (mean, std)
      - all_evals: dict[file] = { query: {metric: value, ...}, ... }
    """
    ndcg_scores = {}
    mrr_scores = {}
    prec_scores = {}
    rec_scores = {}
    all_evals = {}

    # Updated path structure - no more retrieval subfolder
    base = os.path.join(parent_dir, f"qrels/{task.lower()}/{directory}")
    if not os.path.isdir(base):
        logger.warning(f"Skipping missing directory: {base}")
        return ndcg_scores, mrr_scores, prec_scores, rec_scores, all_evals

    json_files = [f for f in sorted(os.listdir(base)) if f.endswith(".json")]
    
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, len(json_files))
    ) as executor:
        futures = [
            executor.submit(
                evaluate_file, task, directory, fname, dataset_cache, top_k
            )
            for fname in json_files
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            fn = result['fname']
            ndcg_scores[fn] = result['ndcg']
            mrr_scores[fn]  = result['mrr']
            prec_scores[fn] = result['prec']
            rec_scores[fn]  = result['rec']
            all_evals[fn]   = result['eval_res']

    return ndcg_scores, mrr_scores, prec_scores, rec_scores, all_evals


async def evaluate_pipeline_mode(
    task, pipeline, mode, dataset_cache, results_queue, top_k
):
    """Evaluate a specific pipeline/mode combination asynchronously"""
    subdir = f"{pipeline}-{mode}"  if mode else pipeline

    ndcg, mrr, prec, rec, all_evals = evaluate_pipeline(
        task, subdir, dataset_cache, top_k
    )
    
    # Prepare summary records
    records = []
    for fn, (mean_ndcg, std_ndcg) in ndcg.items():
        records.append({
            'Pipeline': pipeline,
            'Mode': mode,
            'File': fn,
            f'nDCG@{top_k}': mean_ndcg,
            f'nDCG@{top_k}_std': std_ndcg,
            f'MRR@{top_k}': mrr[fn][0],
            f'MRR@{top_k}_std': mrr[fn][1],
            f'Precision@{top_k}': prec[fn][0],
            f'Precision@{top_k}_std': prec[fn][1],
            f'Recall@{top_k}': rec[fn][0],
            f'Recall@{top_k}_std': rec[fn][1],
        })
    
    # Save per-query metrics (nDCG, MRR, Precision, Recall)
    if all_evals:
        pq_dir = os.path.join(
            parent_dir,
            f"qrels/{task.lower()}/{subdir}/per_query"
        )
        os.makedirs(pq_dir, exist_ok=True)
        for fn, ev in all_evals.items():
            pq = {
                q: {
                    'ndcg': v['ndcg'],
                    'mrr': v['recip_rank'],
                    'precision': v[f'P_{top_k}'],
                    'recall': v[f'recall_{top_k}']
                }
                for q, v in ev.items()
            }
            outpath = os.path.join(pq_dir, f"{fn}_metrics_per_query_top{top_k}.json")
            with open(outpath, "w") as f:
                json.dump(pq, f, indent=4)
        logger.info(f"Saved per-query metrics (nDCG, MRR, Precision, Recall) for {pipeline}-{mode}")
    
    # Prepare for significance testing - group by query column
    ratings_by_query_column = {}
    if all_evals:
        for fn, ev in all_evals.items():
            query_column = os.path.splitext(fn)[0]  # Remove .json extension
            qs = sorted(ev.keys())
            key = f"{pipeline}-{mode}"
            
            if query_column not in ratings_by_query_column:
                ratings_by_query_column[query_column] = {}
            
            ratings_by_query_column[query_column][key] = {
                f'nDCG@{top_k}':     [ev[q]['ndcg'] for q in qs],
                f'MRR@{top_k}':      [ev[q]['recip_rank'] for q in qs],
                f'Precision@{top_k}':[ev[q][f'P_{top_k}'] for q in qs],
                f'Recall@{top_k}':   [ev[q][f'recall_{top_k}'] for q in qs],
            }
    
    await results_queue.put({
        'records': records,
        'ratings_by_query_column': ratings_by_query_column
    })


async def async_main(task: str, top_k: int = 5):
    # Discover available pipelines dynamically from directory structure
    task_dir = os.path.join(parent_dir, f"qrels/{task.lower()}")
    if not os.path.exists(task_dir):
        logger.error(f"Task directory {task_dir} does not exist")
        return
    
    available_dirs = [d for d in os.listdir(task_dir) 
                     if os.path.isdir(os.path.join(task_dir, d)) and not d.startswith('.')]
    
    logger.info(f"Found directories for {task}: {available_dirs}")
    
    # Parse pipeline-mode combinations from directory names
    pipeline_modes = []
    for dir_name in available_dirs:
        parts = dir_name.split('-')
        if len(parts) >= 2:
            pipeline = parts[0]
            mode = '-'.join(parts[1:])  # Handle multi-part modes
            pipeline_modes.append((pipeline, mode))
        else:
            # Handle single word directories as pipeline with no mode
            pipeline_modes.append((parts[0], None))
    
    # Load once, cached forever
    dataset_cache = load_dataset_cache(task)
    logger.info(f"Loaded dataset for {task}")
    
    results_queue = asyncio.Queue()
    eval_tasks    = []
    
    for pipeline, mode in pipeline_modes:
        eval_tasks.append(
            asyncio.create_task(
                evaluate_pipeline_mode(
                    task, pipeline, mode, dataset_cache, results_queue, top_k
                )
            )
        )
    
    records = []
    ratings_by_query_column = {}
    
    for _ in eval_tasks:
        result = await results_queue.get()
        records.extend(result['records'])
        
        # Merge ratings by query column
        for query_column, ratings in result['ratings_by_query_column'].items():
            if query_column not in ratings_by_query_column:
                ratings_by_query_column[query_column] = {}
            ratings_by_query_column[query_column].update(ratings)
    
    await asyncio.gather(*eval_tasks)
    
    # Save a summary Excel - updated path
    results_df = pd.DataFrame(records)
    out_dir    = os.path.join(parent_dir, f"qrels/{task.lower()}")
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_excel(
        os.path.join(out_dir, f"retrieval_results_top{top_k}.xlsx"),
        index=False
    )
    logger.info(f"Saved retrieval_results_top{top_k}.xlsx for {task}")

    # Run iterative significance tests for each query column
    for query_column, ratings in ratings_by_query_column.items():
        if not ratings:  # Skip if no data for this query column
            continue
            
        logger.info(f"Running significance tests for query column: {query_column}")
        
        all_sig_results = []
        remaining_models = list(ratings.keys())
        
        round_num = 1
        while len(remaining_models) > 1:  
            try:
                logger.info(f"Query column '{query_column}' - Round {round_num} with {len(remaining_models)} models")
                
                current_ratings = {model: ratings[model] for model in remaining_models}
                
                sig_df = compare_models(pd.DataFrame.from_dict(current_ratings, orient='index'))
                
                # Add round and query column information
                sig_df['Round'] = round_num
                sig_df['Query_Column'] = query_column
                all_sig_results.append(sig_df)
                
                best_model = None
                highest_score = -1
                
                for model in remaining_models:
                    mean_ndcg = np.mean(current_ratings[model][f'nDCG@{top_k}'])
                    if mean_ndcg > highest_score:
                        highest_score = mean_ndcg
                        best_model = model

                logger.info(f"Query column '{query_column}' - Round {round_num}: Removing best model '{best_model}' with nDCG@{top_k}={highest_score:.4f}")
                
                # Remove the best model for the next round
                remaining_models.remove(best_model)
                round_num += 1
                
            except Exception as e:
                logger.error(f"Error in significance test round {round_num} for query column '{query_column}': {e}")
                break
        
        logger.info(f"Query column '{query_column}': Final 2 models remaining: {remaining_models}")
        
        # Save significance test results for this query column
        if all_sig_results:
            combined_sig_df = pd.concat(all_sig_results, ignore_index=True)
            sig_filename = f"iterative_significance_tests_{query_column}_top{top_k}.xlsx"
            combined_sig_df.to_excel(
                os.path.join(out_dir, sig_filename),
                index=False
            )
            logger.info(f"Saved {sig_filename} for {task}")
    
    logger.info(f"Completed all significance tests for {task}")


def main(task: str, top_k: int = 10):
    """Wrapper to run the async main function"""
    asyncio.run(async_main(task, top_k))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate retrieval pipelines')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to evaluate (default: 5)')
    args = parser.parse_args()

    current_dir = os.path.dirname(__file__)
    parent_dir  = os.path.dirname(current_dir)  # Fixed: only go up one level
    for t in ["FinSlides", "FinReport", "FinQA", "ConvFinQA", "VQAonBD", "TATDQA", "ArxivQA","DUDE","MP-DocVQA","Wiki-ss","SciQAG"]:
        main(t, args.top_k)
