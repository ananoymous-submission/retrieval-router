import os
import json
import asyncio
import time
import logging
import argparse
from tqdm.asyncio import tqdm
from langsmith import traceable
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
from qdrant_client.http import models
from data_collection.pipelines import database_mapping, embedder_mapping, TASK
from data_collection.dataset_loader import load_dataset_for_benchmark

# Set up logging for error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleQueryState(TypedDict):
    """State for single query processing workflow"""
    query: str
    filename: str
    route: str
    embedder: Any
    database: Any
    query_embedding: Any
    retrieved_results: Any
    results_dict: Dict[str, float]
    processed: bool
    embed_latency: float
    search_latency: float
    error: str  # Added to track errors
    failed: bool  # Added to track failure status


class SingleRetrievalTask:
    """Handles retrieval for a single query with a single route/database"""

    def __init__(
        self,
        router = None,
        database_mapping = None,
        embedder_map = None,
        semaphore = None,
        progress_bar = None,
        enable_filtering: bool = False
    ):
        self.router = router
        self.database_mapping = database_mapping
        self.embedder_map = embedder_map
        self.semaphore = semaphore
        self.progress_bar = progress_bar
        self.enable_filtering = enable_filtering

    def create_query_graph(self):
        """Create a LangGraph for processing a single query retrieval"""
        
        def start_processing(state: SingleQueryState) -> SingleQueryState:
            state["query_embedding"] = None
            state["route"] = None
            state["embedder"] = None
            state["database"] = None    
            state["retrieved_results"] = None
            state["results_dict"] = {}
            state["processed"] = False
            state["embed_latency"] = 0.0
            state["search_latency"] = 0.0
            state["error"] = ""
            state["failed"] = False
            return state

        async def classify_query(state: SingleQueryState) -> SingleQueryState:
            """Classify the query into a route"""
            try:
                if self.router is None:
                    route = os.getenv("PIPELINE_NAME").lower()
                else:
                    route = self.router(state["query"])
                    
                state["route"] = route
                state["embedder"] = self.embedder_map[route]
                state["database"] = self.database_mapping[route]
                return state
            except Exception as e:
                logger.error(f"Error in classify_query for query '{state['query'][:50]}...': {str(e)}")
                state["error"] = f"Classification error: {str(e)}"
                state["failed"] = True
                return state

        async def embed_query(state: SingleQueryState) -> SingleQueryState:
            """Embed the query using the embedder"""
            if state.get("failed", False):
                return state
                
            try:
                start_time = time.time()
                query_embedding = await state["embedder"].embed_query(state["query"])
                end_time = time.time()
                
                state["query_embedding"] = query_embedding
                state["embed_latency"] = end_time - start_time
                return state
            except Exception as e:
                logger.error(f"Error in embed_query for query '{state['query'][:50]}...': {str(e)}")
                state["error"] = f"Embedding error: {str(e)}"
                state["failed"] = True
                return state

        async def search_database(state: SingleQueryState) -> SingleQueryState:
            """Search the database with the query embedding"""
            if state.get("failed", False):
                return state

            try:
                start_time = time.time()

                # Build filter conditions if filtering is enabled and filename is available
                filter_conditions = None
                if self.enable_filtering and state.get("filename"):
                    # Extract document_name from filename by splitting on "_" and taking first part
                    document_name = state["filename"].split("_")[0]
                    filter_conditions = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_name",
                                match=models.MatchValue(value=document_name)
                            )
                        ]
                    )

                retrieved = await state["database"].search(
                    state["query_embedding"],
                    filter_conditions=filter_conditions
                )
                end_time = time.time()

                state["retrieved_results"] = retrieved
                state["search_latency"] = end_time - start_time
                return state
            except Exception as e:
                logger.error(f"Error in search_database for query '{state['query'][:50]}...': {str(e)}")
                state["error"] = f"Search error: {str(e)}"
                state["failed"] = True
                return state

        def process_results(state: SingleQueryState) -> SingleQueryState:
            """Process search results into results dictionary"""
            if state.get("failed", False):
                state["results_dict"] = {}
                state["processed"] = False
                return state
                
            try:
                results_dict = {
                    result.metadata["filename"]: result.score 
                    for result in state["retrieved_results"].results
                }
                state["results_dict"] = results_dict
                state["processed"] = True
                return state
            except Exception as e:
                logger.error(f"Error in process_results for query '{state['query'][:50]}...': {str(e)}")
                state["error"] = f"Results processing error: {str(e)}"
                state["failed"] = True
                state["results_dict"] = {}
                state["processed"] = False
                return state

        def finish_processing(state: SingleQueryState) -> SingleQueryState:
            """Complete query processing"""
            if self.progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({"query": state["query"][:50] + "..." if len(state["query"]) > 50 else state["query"]})
            return state

        # Create the graph
        workflow = StateGraph(SingleQueryState)
        
        # Add nodes
        workflow.add_node("start_processing", start_processing)
        workflow.add_node("classify_query", classify_query)
        workflow.add_node("embed_query", embed_query)
        workflow.add_node("search_database", search_database)
        workflow.add_node("process_results", process_results)
        workflow.add_node("finish_processing", finish_processing)
        
        # Add edges
        workflow.add_edge(START, "start_processing")
        workflow.add_edge("start_processing", "classify_query")
        workflow.add_edge("classify_query", "embed_query")
        workflow.add_edge("embed_query", "search_database")
        workflow.add_edge("search_database", "process_results")
        workflow.add_edge("process_results", "finish_processing")
        workflow.add_edge("finish_processing", END)
        
        return workflow.compile()

    @traceable(name="process_query", tags=["query_processing"])
    async def process_query(self, query: str, filename: str = "") -> Dict[str, Any]:
        """Process a single query using LangGraph"""
        async with self.semaphore:
            try:
                # Create the graph for this query
                graph = self.create_query_graph()

                # Initialize state
                initial_state = SingleQueryState(
                    query=query,
                    filename=filename,
                    route="",
                    embedder=None,
                    database=None,
                    query_embedding=None,
                    retrieved_results=None,
                    results_dict={},
                    processed=False,
                    embed_latency=0.0,
                    search_latency=0.0,
                    error="",
                    failed=False
                )
                
                # Execute the graph
                final_state = await graph.ainvoke(initial_state)
                
                # Check if the query failed
                if final_state.get("failed", False):
                    logger.warning(f"Query failed: '{query[:50]}...' - {final_state.get('error', 'Unknown error')}")
                    return {
                        "selected_route": final_state.get("route", "unknown"),
                        "results": {},
                        "embed_latency": final_state["embed_latency"],
                        "search_latency": final_state["search_latency"],
                        "error": final_state.get("error", "Unknown error"),
                        "failed": True
                    }

                return {
                    "selected_route": final_state["route"],
                    "results": final_state["results_dict"],
                    "embed_latency": final_state["embed_latency"],
                    "search_latency": final_state["search_latency"],
                    "error": None,
                    "failed": False
                }
                
            except Exception as e:
                logger.error(f"Unexpected error processing query '{query[:50]}...': {str(e)}")
                return {
                    "selected_route": "unknown",
                    "results": {},
                    "embed_latency": 0.0,
                    "search_latency": 0.0,
                    "error": f"Unexpected error: {str(e)}",
                    "failed": True
                }


class BaseRetrieval():
    def __init__(
        self,
        enable_filtering: bool = False,
        **kwargs
    ):
        self.semaphore = asyncio.Semaphore(100)
        self.hf_token = os.getenv("HF_TOKEN")
        self.qrels_file_base = os.path.join(os.getcwd(), f"qrels/{TASK}/{os.getenv('PIPELINE_NAME')}/")
        self.database_mapping = database_mapping
        self.embedder_map = embedder_mapping
        self.enable_filtering = enable_filtering

        os.makedirs(self.qrels_file_base, exist_ok=True)

    async def process_all_queries(self, query_column: str, batch_size: int = 100):
        """Process all queries from the specified query column using parallel SingleRetrievalTask instances"""
        self.qrels_file = os.path.join(self.qrels_file_base, f"{query_column}.json")

        if os.path.exists(self.qrels_file):
            with open(self.qrels_file, "r") as f:
                self.qrels = json.load(f)
        else:
            with open(self.qrels_file, "w") as f:
                json.dump({}, f, indent=4)
            self.qrels = {}

        dataset = load_dataset_for_benchmark(os.getenv("DATASET"))

        if query_column not in dataset.column_names:
            return

        queries_to_process = []
        for row in dataset:
            if row[query_column] is not None and row[query_column] not in self.qrels.keys():
                # Store both query and filename
                queries_to_process.append({
                    "query": row[query_column],
                    "filename": row.get("image_filename", "")
                })

        # Create progress bar
        with tqdm(total=len(queries_to_process), desc=f"Processing {query_column} queries", unit="queries") as pbar:
            # Create SingleRetrievalTask
            retrieval_task = SingleRetrievalTask(
                database_mapping=self.database_mapping,
                embedder_map=self.embedder_map,
                semaphore=self.semaphore,
                progress_bar=pbar,
                enable_filtering=self.enable_filtering
            )

            # Process queries in batches
            successful_count = 0
            failed_count = 0

            for i in range(0, len(queries_to_process), batch_size):
                batch_queries = queries_to_process[i:i+batch_size]

                # Create tasks for this batch
                batch_tasks = []
                for query_data in batch_queries:
                    task = retrieval_task.process_query(
                        query_data["query"],
                        filename=query_data["filename"]
                    )
                    batch_tasks.append((query_data["query"], task))

                # Execute batch with error handling
                if batch_tasks:
                    # Use return_exceptions=True to prevent one failure from stopping all tasks
                    results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)

                    # Process results, only adding successful tasks to qrels
                    for (query, _), result in zip(batch_tasks, results):
                        if isinstance(result, Exception):
                            # Handle task that failed with an exception - skip adding to qrels
                            logger.error(f"Task failed for query '{query[:50]}...': {str(result)}")
                            failed_count += 1
                        else:
                            # Handle successful task
                            if result.get("failed", False):
                                # Skip failed queries - don't add to qrels
                                logger.warning(f"Skipping failed query '{query[:50]}...': {result.get('error', 'Unknown error')}")
                                failed_count += 1
                            else:
                                # Only add successful queries to qrels
                                self.qrels[query] = result
                                successful_count += 1

                    # Write results to file after each batch
                    try:
                        with open(self.qrels_file, "w") as f:
                            json.dump(self.qrels, f, indent=4)
                    except Exception as e:
                        logger.error(f"Error writing results to file {self.qrels_file}: {str(e)}")

            # Log summary
            logger.info(f"Processed {len(queries_to_process)} queries for {query_column}: {successful_count} successful, {failed_count} failed")

    @traceable(name=f"retrieval_pipeline", tags=["retrieval", TASK, os.getenv("PIPELINE_NAME")])
    async def __call__(self):
        """Main entry point to run the retrieval process."""
        try:
            # Initialize database
            logger.info("Initializing databases...")
            for database_name, database in self.database_mapping.items():
                try:
                    await database.initialize_collection()
                    logger.info(f"Successfully initialized database: {database_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize database {database_name}: {str(e)}")
                    raise e  # Database initialization is critical, so we raise the error
            
            # Process all query columns
            query_columns = ["query"]
            for query_column in query_columns:
                try:
                    logger.info(f"Processing query column: {query_column}")
                    await self.process_all_queries(query_column)
                    logger.info(f"Completed processing query column: {query_column}")
                except Exception as e:
                    logger.error(f"Error processing query column {query_column}: {str(e)}")
                    # Continue with next query column instead of stopping entire pipeline
                    continue
                    
        except Exception as e:
            logger.error(f"Critical error in retrieval pipeline: {str(e)}")
            raise e 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval pipeline with optional document name filtering")
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter documents by document name extracted from the query's associated filename"
    )
    args = parser.parse_args()

    retrieval = BaseRetrieval(enable_filtering=args.filter)
    asyncio.run(retrieval())