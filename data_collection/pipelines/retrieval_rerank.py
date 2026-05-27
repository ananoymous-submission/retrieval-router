import os
import json
import asyncio
import time
import logging
import argparse
from tqdm.asyncio import tqdm
from langsmith import traceable
from typing import Dict, Any, TypedDict, List
from langgraph.graph import StateGraph, START, END
from qdrant_client.http import models
from data_collection.pipelines import database_mapping, embedder_mapping, TASK
from data_collection.dataset_loader import load_dataset_for_benchmark

# Set up logging for error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageQueryState(TypedDict):
    """State for two-stage query processing workflow"""
    query: str
    filename: str
    # Stage 1 - Dense retrieval
    dense_embedder: Any
    dense_database: Any
    dense_embedding: Any
    dense_results: Any
    dense_filenames: List[str]
    retrieve_latency: float
    # Stage 2 - Late reranking
    late_embedder: Any
    late_database: Any
    late_embedding: Any
    rerank_results: Any
    rerank_latency: float
    # Final
    total_latency: float
    results_dict: Dict[str, float]
    dense_results_count: int
    processed: bool
    error: str
    failed: bool


class TwoStageRetrievalTask:
    """Handles two-stage retrieval: dense retrieval followed by late interaction reranking"""

    def __init__(
        self,
        database_mapping: Dict[str, Any],
        embedder_map: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        progress_bar: Any,
        enable_filtering: bool = False,
        dense_limit: int = 100,
        rerank_limit: int = 5,
        mode: str = "multimodal"
    ):
        self.database_mapping = database_mapping
        self.embedder_map = embedder_map
        self.semaphore = semaphore
        self.progress_bar = progress_bar
        self.enable_filtering = enable_filtering
        self.dense_limit = dense_limit
        self.rerank_limit = rerank_limit
        self.mode = mode

        # Set pipeline keys based on mode
        if mode == "multimodal":
            self.dense_key = "multimodal-single"
            self.late_key = "multimodal-multi"
        elif mode == "text":
            self.dense_key = "text-single"
            self.late_key = "text-multi"
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'multimodal' or 'text'")

    def create_query_graph(self):
        """Create a LangGraph for processing a two-stage query retrieval"""

        def start_processing(state: TwoStageQueryState) -> TwoStageQueryState:
            state["dense_embedding"] = None
            state["dense_results"] = None
            state["dense_filenames"] = []
            state["late_embedding"] = None
            state["rerank_results"] = None
            state["results_dict"] = {}
            state["dense_results_count"] = 0
            state["processed"] = False
            state["retrieve_latency"] = 0.0
            state["rerank_latency"] = 0.0
            state["total_latency"] = 0.0
            state["error"] = ""
            state["failed"] = False
            # Set embedders and databases based on mode
            state["dense_embedder"] = self.embedder_map[self.dense_key]
            state["dense_database"] = self.database_mapping[self.dense_key]
            state["late_embedder"] = self.embedder_map[self.late_key]
            state["late_database"] = self.database_mapping[self.late_key]
            return state

        async def embed_dense(state: TwoStageQueryState) -> TwoStageQueryState:
            """Embed the query using the dense embedder (Stage 1)"""
            if state.get("failed", False):
                return state

            try:
                start_time = time.time()
                dense_embedding = await state["dense_embedder"].embed_query(state["query"])
                state["dense_embedding"] = dense_embedding
                state["retrieve_latency"] = time.time() - start_time
                return state
            except Exception as e:
                logger.error(f"Error in embed_dense for query '{state['query'][:50]}...': {str(e)}")
                state["error"] = f"Dense embedding error: {str(e)}"
                state["failed"] = True
                return state

        async def search_dense(state: TwoStageQueryState) -> TwoStageQueryState:
            """Search the dense database (Stage 1)"""
            if state.get("failed", False):
                return state

            try:
                start_time = time.time()

                # Build filter conditions if filtering is enabled and filename is available
                filter_conditions = None
                if self.enable_filtering and state.get("filename"):
                    document_name = state["filename"].split("_")[0]
                    filter_conditions = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_name",
                                match=models.MatchValue(value=document_name)
                            )
                        ]
                    )

                retrieved = await state["dense_database"].search(
                    state["dense_embedding"],
                    limit=self.dense_limit,
                    filter_conditions=filter_conditions
                )

                state["dense_results"] = retrieved
                state["dense_filenames"] = [
                    result.metadata["filename"]
                    for result in retrieved.results
                ]
                state["dense_results_count"] = len(state["dense_filenames"])
                state["retrieve_latency"] += time.time() - start_time
                return state
            except Exception as e:
                logger.error(f"Error in search_dense for query '{state['query'][:50]}...': {str(e)}")
                state["error"] = f"Dense search error: {str(e)}"
                state["failed"] = True
                return state

        async def embed_late(state: TwoStageQueryState) -> TwoStageQueryState:
            """Embed the query using the late interaction embedder (Stage 2)"""
            if state.get("failed", False):
                return state

            # Skip if no results from dense retrieval
            if not state["dense_filenames"]:
                state["error"] = "No results from dense retrieval to rerank"
                state["failed"] = True
                return state

            try:
                start_time = time.time()
                late_embedding = await state["late_embedder"].embed_query(state["query"])
                state["late_embedding"] = late_embedding
                state["rerank_latency"] = time.time() - start_time
                return state
            except Exception as e:
                logger.error(f"Error in embed_late for query '{state['query'][:50]}...': {str(e)}")
                state["error"] = f"Late embedding error: {str(e)}"
                state["failed"] = True
                return state

        async def search_late_rerank(state: TwoStageQueryState) -> TwoStageQueryState:
            """Search the late interaction database with filename filter (Stage 2)"""
            if state.get("failed", False):
                return state

            try:
                start_time = time.time()

                # Search with filter by filenames from dense retrieval
                retrieved = await state["late_database"].search(
                    state["late_embedding"],
                    limit=self.rerank_limit,
                    filter_by_filenames=state["dense_filenames"]
                )

                state["rerank_results"] = retrieved
                state["rerank_latency"] += time.time() - start_time
                return state
            except Exception as e:
                logger.error(f"Error in search_late_rerank for query '{state['query'][:50]}...': {str(e)}")
                state["error"] = f"Late search/rerank error: {str(e)}"
                state["failed"] = True
                return state

        def process_results(state: TwoStageQueryState) -> TwoStageQueryState:
            """Process reranking results into results dictionary"""
            if state.get("failed", False):
                state["results_dict"] = {}
                state["processed"] = False
                return state

            try:
                results_dict = {
                    result.metadata["filename"]: result.score
                    for result in state["rerank_results"].results
                }
                state["results_dict"] = results_dict
                state["total_latency"] = state["retrieve_latency"] + state["rerank_latency"]
                state["processed"] = True
                return state
            except Exception as e:
                logger.error(f"Error in process_results for query '{state['query'][:50]}...': {str(e)}")
                state["error"] = f"Results processing error: {str(e)}"
                state["failed"] = True
                state["results_dict"] = {}
                state["processed"] = False
                return state

        def finish_processing(state: TwoStageQueryState) -> TwoStageQueryState:
            """Complete query processing"""
            if self.progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({
                    "query": state["query"][:30] + "..." if len(state["query"]) > 30 else state["query"]
                })
            return state

        # Create the graph
        workflow = StateGraph(TwoStageQueryState)

        # Add nodes
        workflow.add_node("start_processing", start_processing)
        workflow.add_node("embed_dense", embed_dense)
        workflow.add_node("search_dense", search_dense)
        workflow.add_node("embed_late", embed_late)
        workflow.add_node("search_late_rerank", search_late_rerank)
        workflow.add_node("process_results", process_results)
        workflow.add_node("finish_processing", finish_processing)

        # Add edges
        workflow.add_edge(START, "start_processing")
        workflow.add_edge("start_processing", "embed_dense")
        workflow.add_edge("embed_dense", "search_dense")
        workflow.add_edge("search_dense", "embed_late")
        workflow.add_edge("embed_late", "search_late_rerank")
        workflow.add_edge("search_late_rerank", "process_results")
        workflow.add_edge("process_results", "finish_processing")
        workflow.add_edge("finish_processing", END)

        return workflow.compile()

    @traceable(name="two_stage_process_query", tags=["query_processing", "two_stage"])
    async def process_query(self, query: str, filename: str = "") -> Dict[str, Any]:
        """Process a single query using two-stage LangGraph"""
        async with self.semaphore:
            try:
                # Create the graph for this query
                graph = self.create_query_graph()

                # Initialize state
                initial_state = TwoStageQueryState(
                    query=query,
                    filename=filename,
                    dense_embedder=None,
                    dense_database=None,
                    dense_embedding=None,
                    dense_results=None,
                    dense_filenames=[],
                    retrieve_latency=0.0,
                    late_embedder=None,
                    late_database=None,
                    late_embedding=None,
                    rerank_results=None,
                    rerank_latency=0.0,
                    total_latency=0.0,
                    results_dict={},
                    dense_results_count=0,
                    processed=False,
                    error="",
                    failed=False
                )

                # Execute the graph
                final_state = await graph.ainvoke(initial_state)

                # Check if the query failed
                if final_state.get("failed", False):
                    logger.warning(f"Query failed: '{query[:50]}...' - {final_state.get('error', 'Unknown error')}")
                    return {
                        "selected_route": "two-stage",
                        "results": {},
                        "retrieve_latency": final_state["retrieve_latency"],
                        "rerank_latency": final_state["rerank_latency"],
                        "total_latency": final_state["total_latency"],
                        "dense_results_count": final_state["dense_results_count"],
                        "error": final_state.get("error", "Unknown error"),
                        "failed": True
                    }

                return {
                    "selected_route": "two-stage",
                    "results": final_state["results_dict"],
                    "retrieve_latency": final_state["retrieve_latency"],
                    "rerank_latency": final_state["rerank_latency"],
                    "total_latency": final_state["total_latency"],
                    "dense_results_count": final_state["dense_results_count"],
                    "error": None,
                    "failed": False
                }

            except Exception as e:
                logger.error(f"Unexpected error processing query '{query[:50]}...': {str(e)}")
                return {
                    "selected_route": "two-stage",
                    "results": {},
                    "retrieve_latency": 0.0,
                    "rerank_latency": 0.0,
                    "total_latency": 0.0,
                    "dense_results_count": 0,
                    "error": f"Unexpected error: {str(e)}",
                    "failed": True
                }


class TwoStageRetrieval:
    def __init__(
        self,
        enable_filtering: bool = False,
        dense_limit: int = 100,
        rerank_limit: int = 5,
        mode: str = "multimodal",
        **kwargs
    ):
        self.semaphore = asyncio.Semaphore(20)
        self.hf_token = os.getenv("HF_TOKEN")
        self.mode = mode
        self.qrels_file_base = os.path.join(os.getcwd(), f"qrels/{TASK}/TWO-STAGE-{mode.upper()}/")
        self.database_mapping = database_mapping
        self.embedder_map = embedder_mapping
        self.enable_filtering = enable_filtering
        self.dense_limit = dense_limit
        self.rerank_limit = rerank_limit

        os.makedirs(self.qrels_file_base, exist_ok=True)

    async def process_all_queries(self, query_column: str, batch_size: int = 100):
        """Process all queries from the specified query column using two-stage retrieval"""
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
                queries_to_process.append({
                    "query": row[query_column],
                    "filename": row.get("image_filename", "")
                })

        # Create progress bar
        with tqdm(total=len(queries_to_process), desc=f"Processing {query_column} queries (two-stage)", unit="queries") as pbar:
            # Create TwoStageRetrievalTask
            retrieval_task = TwoStageRetrievalTask(
                database_mapping=self.database_mapping,
                embedder_map=self.embedder_map,
                semaphore=self.semaphore,
                progress_bar=pbar,
                enable_filtering=self.enable_filtering,
                dense_limit=self.dense_limit,
                rerank_limit=self.rerank_limit,
                mode=self.mode
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
                    results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)

                    # Process results, only adding successful tasks to qrels
                    for (query, _), result in zip(batch_tasks, results):
                        if isinstance(result, Exception):
                            logger.error(f"Task failed for query '{query[:50]}...': {str(result)}")
                            failed_count += 1
                        else:
                            if result.get("failed", False):
                                logger.warning(f"Skipping failed query '{query[:50]}...': {result.get('error', 'Unknown error')}")
                                failed_count += 1
                            else:
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

    @traceable(name="two_stage_retrieval_pipeline", tags=["retrieval", "two_stage", TASK])
    async def __call__(self):
        """Main entry point to run the two-stage retrieval process."""
        try:
            # Initialize databases
            logger.info("Initializing databases for two-stage retrieval...")
            for database_name, database in self.database_mapping.items():
                try:
                    await database.initialize_collection()
                    logger.info(f"Successfully initialized database: {database_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize database {database_name}: {str(e)}")
                    raise e

            # Process all query columns
            query_columns = ["query"]
            for query_column in query_columns:
                try:
                    logger.info(f"Processing query column: {query_column}")
                    await self.process_all_queries(query_column)
                    logger.info(f"Completed processing query column: {query_column}")
                except Exception as e:
                    logger.error(f"Error processing query column {query_column}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Critical error in two-stage retrieval pipeline: {str(e)}")
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run two-stage retrieval pipeline: dense retrieval + late interaction reranking")
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter documents by document name extracted from the query's associated filename"
    )
    parser.add_argument(
        "--dense-limit",
        type=int,
        default=100,
        help="Number of documents to retrieve in Stage 1 (dense retrieval). Default: 100"
    )
    parser.add_argument(
        "--rerank-limit",
        type=int,
        default=5,
        help="Number of documents to return after Stage 2 (reranking). Default: 5"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["multimodal", "text"],
        default="multimodal",
        help="Pipeline mode: 'multimodal' (dense=multimodal-single, late=multimodal-multi) or 'text' (dense=text-single, late=text-multi). Default: multimodal"
    )
    args = parser.parse_args()

    retrieval = TwoStageRetrieval(
        enable_filtering=args.filter,
        dense_limit=args.dense_limit,
        rerank_limit=args.rerank_limit,
        mode=args.mode
    )
    asyncio.run(retrieval())
