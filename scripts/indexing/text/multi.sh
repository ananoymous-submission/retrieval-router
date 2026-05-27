python -c "from dotenv import load_dotenv; load_dotenv('.env')"

export PIPELINE_NAME="TEXT-MULTI"
export TEXT_MULTI_EMBEDDER="colbert"
export TEXT_MULTI_VECTOR_SIZE=128

export TASK=finreport
export DATASET=X/FinReport

python -m data_collection.pipelines.indexing

export TASK=finslides
export DATASET=X/FinSlides

python -m data_collection.pipelines.indexing

export TASK=finqa
export DATASET=X/FinQA

python -m data_collection.pipelines.indexing

export TASK=convfinqa
export DATASET=X/ConvFinQA

python -m data_collection.pipelines.indexing

export TASK=vqaonbd
export DATASET=X/VQAonBD

python -m data_collection.pipelines.indexing

export TASK=tatdqa
export DATASET=X/TATDQA

python -m data_collection.pipelines.indexing

export TASK=arxivqa
export DATASET=X/ArxivQA

python -m data_collection.pipelines.indexing 

export TASK=mp-docvqa
export DATASET=X/MP-DocVQA

python -m data_collection.pipelines.indexing 

export TASK=mp-sciqag
export DATASET=X/SciQAG

python -m data_collection.pipelines.indexing 

export TASK=dude
export DATASET=X/DUDE

python -m data_collection.pipelines.indexing 

export TASK=wiki-ss
export DATASET=X/Wiki-ss

python -m data_collection.pipelines.indexing 