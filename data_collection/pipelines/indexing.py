import os
import datetime
import asyncio
from tqdm.asyncio import tqdm
from langsmith import traceable
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from data_collection.pipelines import database_mapping, embedder_mapping, TASK
from data_collection.dataset_loader import load_dataset_for_benchmark
from data_collection.vector_dabases import BaseVectorDatabase

class SingleDocumentState(TypedDict):
    """State for single document processing workflow"""
    filename: str
    image: Any
    text: Any
    metadata: Dict[str, Any]
    embedding: Any
    processed: bool 

class SingleIndexingTask:
    """Handles indexing for a single route/database combination"""
    
    def __init__(   
        self,
        route: str,
        database: BaseVectorDatabase,
        embedder: Any,
        embedding_semaphore: asyncio.Semaphore = None,
        upserting_semaphore: asyncio.Semaphore = None,
        progress_bar: tqdm = None
    ):
        self.route = route
        self.database = database
        self.embedder = embedder
        self.embedding_semaphore = embedding_semaphore
        self.upserting_semaphore = upserting_semaphore
        self.progress_bar = progress_bar

    def create_document_graph(self):
        """Create a LangGraph for processing a single document with single route"""
        
        def start_processing(state: SingleDocumentState) -> SingleDocumentState:
            """Initialize document processing"""
            state["metadata"] = {"filename": state["filename"], "document_name" : state["filename"].split("_")[0]}
            state["embedding"] = None
            state["processed"] = False
            return state

        @traceable(name="embed_document")
        async def embed_document(state: SingleDocumentState) -> SingleDocumentState:
            """Embed the document using the embedder"""
            async with self.embedding_semaphore:
                # Use text for TEXT routes, image for MULTIMODAL routes
                document_content = state["text"] if self.route.upper().startswith("TEXT") else state["image"]
                embedding = await self.embedder.embed_document(document_content)
                state["embedding"] = embedding
            return state

        @traceable(name="upsert_document")
        async def upsert_document(state: SingleDocumentState) -> SingleDocumentState:
            """Upsert the embedding to the database"""
            embedding = state["embedding"]
            metadata = state["metadata"]

            async with self.upserting_semaphore:
                await self.database.index_document(embedding, metadata)

            state["processed"] = True
            return state

        def finish_processing(state: SingleDocumentState) -> SingleDocumentState:
            """Complete document processing"""
            if self.progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({"file": state["filename"]})
            return state

        workflow = StateGraph(SingleDocumentState)

        workflow.add_node("start_processing", start_processing)
        workflow.add_node("embed_document", embed_document)
        workflow.add_node("upsert_document", upsert_document)
        workflow.add_node("finish_processing", finish_processing)

        workflow.add_edge(START, "start_processing")
        workflow.add_edge("start_processing", "embed_document")
        workflow.add_edge("embed_document", "upsert_document")
        workflow.add_edge("upsert_document", "finish_processing")
        workflow.add_edge("finish_processing", END)

        return workflow.compile()

    @traceable(name="process_document", tags=["single_indexing_task"])
    async def process_document(self, filename: str, image: Any, text: Any) -> None:
        """Process a single document using LangGraph"""
        graph = self.create_document_graph()

        try:
            initial_state = SingleDocumentState(
                filename=filename,
                image=image,
                text=text,
                metadata={},
                embedding=None,
                processed=False
            )

            final_state = await graph.ainvoke(initial_state)
            return final_state

        except Exception as e:
            print(f"Error processing document {filename}: {e}")
            return None
            
class BaseIndexing:
    def __init__(
        self,
        **kwargs
    ):
        self.embedding_semaphore = asyncio.Semaphore(10)
        self.upserting_semaphore = asyncio.Semaphore(10)
        self.embedder_map = embedder_mapping
        self.database_mapping = database_mapping

    async def process_all_files(self, indexed_files: set, batch_size: int = 100):
        """Process all not indexed files from the dataset using a single route"""

        dataset = load_dataset_for_benchmark(os.getenv("DATASET"))

        all_keys = set(dataset["image_filename"])

        route = os.getenv("PIPELINE_NAME").lower()
        database = self.database_mapping[route]

        keys_to_process = all_keys - indexed_files

        total_docs = len(keys_to_process)

        with tqdm(total=total_docs, desc="Processing documents", unit="docs") as pbar:
            if not keys_to_process:
                return

            embedder = self.embedder_map[route]

            indexing_task = SingleIndexingTask(
                route=route,
                database=database,
                embedder=embedder,
                embedding_semaphore=self.embedding_semaphore,
                upserting_semaphore=self.upserting_semaphore,
                progress_bar=pbar
            )

            batch = []
            for row in dataset:
                if row["image_filename"] in keys_to_process:
                    keys_to_process.remove(row["image_filename"])
                    task = indexing_task.process_document(row["image_filename"], row["image"], row["text"])
                    batch.append(task)

                    # Process batch when it reaches batch_size
                    if len(batch) >= batch_size:
                        await asyncio.gather(*batch)
                        batch = []

            # Process remaining tasks
            if batch:
                await asyncio.gather(*batch)

    async def __call__(self):
        """Main method that orchestrates the entire indexing process"""
        
        os.environ["LANGCHAIN_PROJECT"] = f"{TASK}-{os.getenv('PIPELINE_NAME')}-indexing-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        
        route = list(self.database_mapping.keys())[0]
        
        database = self.database_mapping[route]
        await database.initialize_collection()
        
        indexed_files = await database.get_indexed_files()
        await self.process_all_files(indexed_files)

if __name__ == "__main__":
    indexing = BaseIndexing()
    asyncio.run(indexing())