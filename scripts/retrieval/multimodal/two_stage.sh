#!/usr/bin/env bash
set -euo pipefail

HF_NAMESPACE="${HF_NAMESPACE:-emrekuruu}"

export PIPELINE_NAME="TWO-STAGE"
export TWO_STAGE_MODE="multimodal"  # Options: "multimodal" or "text"

# Multimodal mode embedders
export MULTIMODAL_SINGLE_EMBEDDER="biqwen"
export MULTIMODAL_SINGLE_VECTOR_SIZE=3584
export MULTIMODAL_MULTI_EMBEDDER="colqwen"
export MULTIMODAL_MULTI_VECTOR_SIZE=128

# Text mode embedders (uncomment if using TWO_STAGE_MODE="text")
# export TEXT_SINGLE_EMBEDDER="linq"
# export TEXT_SINGLE_VECTOR_SIZE=4096
# export TEXT_MULTI_EMBEDDER="colbert"
# export TEXT_MULTI_VECTOR_SIZE=128

export TASK=finreport
export DATASET=${HF_NAMESPACE}/FinReport

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=finslides
export DATASET=${HF_NAMESPACE}/FinSlides

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=finqa
export DATASET=${HF_NAMESPACE}/FinQA

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=convfinqa
export DATASET=${HF_NAMESPACE}/ConvFinQA

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=vqaonbd
export DATASET=${HF_NAMESPACE}/VQAonBD

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=tatdqa
export DATASET=${HF_NAMESPACE}/TATDQA

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=arxivqa
export DATASET=${HF_NAMESPACE}/ArxivQA

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=mp-docvqa
export DATASET=${HF_NAMESPACE}/MP-DocVQA

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=mp-sciqag
export DATASET=${HF_NAMESPACE}/SciQAG

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=dude
export DATASET=${HF_NAMESPACE}/DUDE

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=wiki-ss
export DATASET=${HF_NAMESPACE}/Wiki-ss

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE
