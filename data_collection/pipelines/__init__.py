import os
from dotenv import load_dotenv
from data_collection.embedders import EmbedderConfig, EmbedderFactory
from data_collection.vector_databases import DatabaseConfig, VectorDatabaseFactory

# Load environment variables from .env file
load_dotenv('.env')

__all__ = ["database_mapping", "embedder_mapping", "TASK"]

TASK = os.getenv("TASK")

PIPELINE_NAME = os.getenv("PIPELINE_NAME")

database_mapping = {}

multimodal_single_database_config = DatabaseConfig(
    url=os.getenv("DATABASE_URL"),
    api_key=os.getenv("DATABASE_API_KEY"),
    collection_name=f"multimodal-single-{TASK}",
    vector_size=int(os.getenv("MULTIMODAL_SINGLE_VECTOR_SIZE")) if os.getenv("MULTIMODAL_SINGLE_VECTOR_SIZE") else -1,
    vector_type="single",
)

multimodal_multi_database_config = DatabaseConfig(
    url=os.getenv("DATABASE_URL"),
    api_key=os.getenv("DATABASE_API_KEY"),
    collection_name=f"multimodal-multi-{TASK}",
    vector_size=int(os.getenv("MULTIMODAL_MULTI_VECTOR_SIZE")) if os.getenv("MULTIMODAL_MULTI_VECTOR_SIZE") else -1,
        vector_type="multi",
    )

text_single_database_config = DatabaseConfig(
    url=os.getenv("DATABASE_URL"),
    api_key=os.getenv("DATABASE_API_KEY"),
    collection_name=f"text-single-{TASK}",
    vector_size=int(os.getenv("TEXT_SINGLE_VECTOR_SIZE")) if os.getenv("TEXT_SINGLE_VECTOR_SIZE") else -1,
    vector_type="single",
)

text_multi_database_config = DatabaseConfig(
    url=os.getenv("DATABASE_URL"),
    api_key=os.getenv("DATABASE_API_KEY"),
    collection_name=f"text-multi-{TASK}",
    vector_size=int(os.getenv("TEXT_MULTI_VECTOR_SIZE")) if os.getenv("TEXT_MULTI_VECTOR_SIZE") else -1,
    vector_type="multi",
    )

if PIPELINE_NAME == "MULTIMODAL-SINGLE":
    database_mapping["multimodal-single"] = VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"),multimodal_single_database_config)
elif PIPELINE_NAME == "MULTIMODAL-MULTI":
    database_mapping["multimodal-multi"] = VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"),multimodal_multi_database_config)
elif PIPELINE_NAME == "TEXT-MULTI":
    database_mapping["text-multi"] = VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"),text_multi_database_config)
elif PIPELINE_NAME == "TEXT-SINGLE":
    database_mapping["text-single"] = VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"),text_single_database_config)
elif PIPELINE_NAME == "BM25":
    database_mapping["bm25"] = None
elif PIPELINE_NAME == "TWO-STAGE":
    TWO_STAGE_MODE = os.getenv("TWO_STAGE_MODE", "multimodal")
    if TWO_STAGE_MODE == "multimodal":
        database_mapping["multimodal-single"] = VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"), multimodal_single_database_config)
        database_mapping["multimodal-multi"] = VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"), multimodal_multi_database_config)
    elif TWO_STAGE_MODE == "text":
        database_mapping["text-single"] = VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"), text_single_database_config)
        database_mapping["text-multi"] = VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"), text_multi_database_config)
else:
    for pipeline in ["multimodal-single", "multimodal-multi", "text-multi", "text-single"]:
        database_mapping[pipeline] = VectorDatabaseFactory.create_database(os.getenv("DATABASE_TYPE"),globals()[f"{pipeline}_database_config"])


if PIPELINE_NAME == "MULTIMODAL-SINGLE":
    multimodal_single_embedder_config = EmbedderConfig(
        model_name=os.getenv("MULTIMODAL_SINGLE_EMBEDDER"),
    )

if PIPELINE_NAME == "MULTIMODAL-MULTI":
    multimodal_multi_embedder_config = EmbedderConfig(
        model_name=os.getenv("MULTIMODAL_MULTI_EMBEDDER"),
    )

if PIPELINE_NAME == "TEXT-MULTI":
    text_multi_embedder_config = EmbedderConfig(
        model_name=os.getenv("TEXT_MULTI_EMBEDDER"),
    )

if PIPELINE_NAME == "TEXT-SINGLE":
    text_single_embedder_config = EmbedderConfig(
        model_name=os.getenv("TEXT_SINGLE_EMBEDDER"),
    )

if PIPELINE_NAME == "TWO-STAGE":
    TWO_STAGE_MODE = os.getenv("TWO_STAGE_MODE", "multimodal")
    if TWO_STAGE_MODE == "multimodal":
        multimodal_single_embedder_config = EmbedderConfig(
            model_name=os.getenv("MULTIMODAL_SINGLE_EMBEDDER"),
        )
        multimodal_multi_embedder_config = EmbedderConfig(
            model_name=os.getenv("MULTIMODAL_MULTI_EMBEDDER"),
        )
    elif TWO_STAGE_MODE == "text":
        text_single_embedder_config = EmbedderConfig(
            model_name=os.getenv("TEXT_SINGLE_EMBEDDER"),
        )
        text_multi_embedder_config = EmbedderConfig(
            model_name=os.getenv("TEXT_MULTI_EMBEDDER"),
        )

embedder_mapping = {}

if PIPELINE_NAME == "MULTIMODAL-SINGLE":
    embedder_mapping["multimodal-single"] = EmbedderFactory.create_embedder(os.getenv("MULTIMODAL_SINGLE_EMBEDDER"), multimodal_single_embedder_config)
elif PIPELINE_NAME == "MULTIMODAL-MULTI":
    embedder_mapping["multimodal-multi"] = EmbedderFactory.create_embedder(os.getenv("MULTIMODAL_MULTI_EMBEDDER"), multimodal_multi_embedder_config)
elif PIPELINE_NAME == "TEXT-MULTI":
    embedder_mapping["text-multi"] = EmbedderFactory.create_embedder(os.getenv("TEXT_MULTI_EMBEDDER"), text_multi_embedder_config)
elif PIPELINE_NAME == "TEXT-SINGLE":
    embedder_mapping["text-single"] = EmbedderFactory.create_embedder(os.getenv("TEXT_SINGLE_EMBEDDER"), text_single_embedder_config)
elif PIPELINE_NAME == "BM25":
    embedder_mapping["bm25"] = None
elif PIPELINE_NAME == "TWO-STAGE":
    TWO_STAGE_MODE = os.getenv("TWO_STAGE_MODE", "multimodal")
    if TWO_STAGE_MODE == "multimodal":
        embedder_mapping["multimodal-single"] = EmbedderFactory.create_embedder(os.getenv("MULTIMODAL_SINGLE_EMBEDDER"), multimodal_single_embedder_config)
        embedder_mapping["multimodal-multi"] = EmbedderFactory.create_embedder(os.getenv("MULTIMODAL_MULTI_EMBEDDER"), multimodal_multi_embedder_config)
    elif TWO_STAGE_MODE == "text":
        embedder_mapping["text-single"] = EmbedderFactory.create_embedder(os.getenv("TEXT_SINGLE_EMBEDDER"), text_single_embedder_config)
        embedder_mapping["text-multi"] = EmbedderFactory.create_embedder(os.getenv("TEXT_MULTI_EMBEDDER"), text_multi_embedder_config)
else:
    for pipeline in ["multimodal-single", "multimodal-multi", "text-multi", "text-single"]:
        embedder_mapping[pipeline] = EmbedderFactory.create_embedder(os.getenv(f"{pipeline.upper()}_EMBEDDER"), globals()[f"{pipeline}_embedder_config"])