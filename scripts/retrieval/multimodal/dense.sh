#!/usr/bin/env bash
set -euo pipefail

HF_NAMESPACE="${HF_NAMESPACE:-emrekuruu}"

export PIPELINE_NAME="MULTIMODAL-SINGLE"
export MULTIMODAL_SINGLE_EMBEDDER="biqwen"
export MULTIMODAL_SINGLE_VECTOR_SIZE=3584

# export TASK=finreport
# export DATASET=${HF_NAMESPACE}/FinReport

# python -m data_collection.pipelines.retrieval

# export TASK=finslides
# export DATASET=${HF_NAMESPACE}/FinSlides

# python -m data_collection.pipelines.retrieval

# export TASK=finqa
# export DATASET=${HF_NAMESPACE}/FinQA

# python -m data_collection.pipelines.retrieval

# export TASK=convfinqa
# export DATASET=${HF_NAMESPACE}/ConvFinQA

# python -m data_collection.pipelines.retrieval

# export TASK=vqaonbd
# export DATASET=${HF_NAMESPACE}/VQAonBD

# python -m data_collection.pipelines.retrieval

# export TASK=tatdqa
# export DATASET=${HF_NAMESPACE}/TATDQA

# python -m data_collection.pipelines.retrieval

export TASK=arxivqa
export DATASET=${HF_NAMESPACE}/ArxivQA

python -m data_collection.pipelines.retrieval 

export TASK=mp-docvqa
export DATASET=${HF_NAMESPACE}/MP-DocVQA

python -m data_collection.pipelines.retrieval

export TASK=mp-sciqag
export DATASET=${HF_NAMESPACE}/SciQAG

python -m data_collection.pipelines.retrieval 

export TASK=dude
export DATASET=${HF_NAMESPACE}/DUDE

python -m data_collection.pipelines.retrieval 

export TASK=wiki-ss
export DATASET=${HF_NAMESPACE}/Wiki-ss

python -m data_collection.pipelines.retrieval