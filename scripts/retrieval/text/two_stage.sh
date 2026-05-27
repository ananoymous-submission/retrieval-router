python -c "from dotenv import load_dotenv; load_dotenv('.env')"

export PIPELINE_NAME="TWO-STAGE"
export TWO_STAGE_MODE="text"  

export TEXT_SINGLE_EMBEDDER="linq"
export TEXT_SINGLE_VECTOR_SIZE=4096
export TEXT_MULTI_EMBEDDER="colbert"
export TEXT_MULTI_VECTOR_SIZE=128

export TASK=finreport
export DATASET=X/FinReport

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=finslides
export DATASET=X/FinSlides

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=finqa
export DATASET=X/FinQA

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=convfinqa
export DATASET=X/ConvFinQA

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=vqaonbd
export DATASET=X/VQAonBD

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=tatdqa
export DATASET=X/TATDQA

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=arxivqa
export DATASET=X/ArxivQA

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=mp-docvqa
export DATASET=X/MP-DocVQA

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=mp-sciqag
export DATASET=X/SciQAG

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=dude
export DATASET=X/DUDE

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE

export TASK=wiki-ss
export DATASET=X/Wiki-ss

python -m data_collection.pipelines.retrieval_rerank --mode $TWO_STAGE_MODE
