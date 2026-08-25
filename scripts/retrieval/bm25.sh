#!/usr/bin/env bash
set -euo pipefail

: "${TASK:?Set TASK to the dataset slug}"
: "${DATASET:?Set DATASET to the Hugging Face dataset ID}"

export PIPELINE_NAME="BM25"
python -m data_collection.pipelines.bm25
