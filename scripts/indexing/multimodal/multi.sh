#!/usr/bin/env bash
set -euo pipefail

HF_NAMESPACE="${HF_NAMESPACE:-emrekuruu}"

export PIPELINE_NAME="MULTIMODAL-MULTI"
export MULTIMODAL_MULTI_EMBEDDER="colqwen"
export MULTIMODAL_MULTI_VECTOR_SIZE=128

export TASK=finreport
export DATASET=${HF_NAMESPACE}/FinReport

python -m data_collection.pipelines.indexing

export TASK=finslides
export DATASET=${HF_NAMESPACE}/FinSlides

python -m data_collection.pipelines.indexing

export TASK=finqa
export DATASET=${HF_NAMESPACE}/FinQA

python -m data_collection.pipelines.indexing

export TASK=convfinqa
export DATASET=${HF_NAMESPACE}/ConvFinQA

python -m data_collection.pipelines.indexing

export TASK=vqaonbd
export DATASET=${HF_NAMESPACE}/VQAonBD

python -m data_collection.pipelines.indexing

export TASK=tatdqa
export DATASET=${HF_NAMESPACE}/TATDQA

python -m data_collection.pipelines.indexing
