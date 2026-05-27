import os
import asyncio
import logging
import time
import json
from abc import ABC, abstractmethod
from data_collection.dataset_loader import load_dataset_for_benchmark
from data_collection.pipelines import TASK
from tqdm import tqdm

class BaseSparseRetrieval():
    """Base class for sparse retrieval methods like BM25 and SPLADE."""
    
    def __init__(
        self,
        top_k: int = 5,
        **kwargs
    ):
        # Don't call super().__init__() to avoid embedder requirements
        self.semaphore = asyncio.Semaphore(120)
        self.hf_token = os.getenv("HF_TOKEN")
        self.qrels_file_base = os.path.join(os.getcwd(), f"qrels/{TASK}/{os.getenv('PIPELINE_NAME', 'SPARSE')}/")
        
        os.makedirs(self.qrels_file_base, exist_ok=True)
        
        self.top_k = top_k
        self.task = TASK
        
        # Load pre-processed chunks
        chunks_path = os.path.join(os.path.dirname(__file__), "..", "preprocessing", "chunks", f"{TASK}_chunks.json")
        with open(chunks_path, "r") as f:
            self.chunks_data = json.load(f)
            
        # Initialize timing logs for performance tracking
        self.timing_logs = []
        
    async def prepare_corpus(self):
        """Prepare the corpus for retrieval using pre-processed chunks."""
        corpus = []
        self.documents = []
        
        print("📚 Loading corpus from pre-processed chunks...")
        
        # Process chunks directly from pre-processed data (much faster!)
        for filename, file_data in tqdm(self.chunks_data.items(), desc="Processing documents"):
            file_chunks = file_data.get("chunks", [])
            
            for chunk in file_chunks:
                corpus.append(chunk)
                self.documents.append({"text": chunk, "filename": filename})
        
        self.corpus = corpus
        logging.info(f"Prepared corpus with {len(corpus)} chunks from {len(self.chunks_data)} documents")
        
        print("🔍 Building retrieval index...")
        await self._prepare_retrieval_model()

    @abstractmethod
    async def _prepare_retrieval_model(self):
        """Prepare the specific retrieval model (BM25, SPLADE, etc.)"""
        pass

    @abstractmethod
    async def search(self, query, top_k=None):
        """Search for documents given a query. Should return dict of {filename: score}"""
        pass

    async def embed_query(self, query):
        """
        For sparse methods, we don't actually need embeddings, but we need to maintain
        the same interface as BaseRetrieval. Return None for both text and visual
        embeddings since sparse methods will handle the query directly.
        """
        return None, None

    async def process_query(self, query):
        """
        Process a single query efficiently - no file I/O during processing.
        """
        total_start_time = time.monotonic()
        
        try:
            async with self.semaphore:
                # Direct sparse search - no embedding needed
                retrieval_start_time = time.monotonic()
                retrieved = await self.search(query, top_k=self.top_k)
                retrieval_duration = time.monotonic() - retrieval_start_time
                
                total_duration = time.monotonic() - total_start_time
                
                self.timing_logs.append({
                    "query": query,
                    "retrieval_duration_sec": retrieval_duration,
                    "total_duration_sec": total_duration,
                    "status": "Success",
                    "results_count": len(retrieved)
                })
                
                return query, retrieved

        except Exception as e:
            logging.error(f"Error during {self.__class__.__name__} retrieval for query {query}: {e}")
            total_duration = time.monotonic() - total_start_time
            self.timing_logs.append({
                "query": query,
                "retrieval_duration_sec": 0,
                "total_duration_sec": total_duration,
                "status": "Error",
                "error": str(e)
            })
            return query, {}

    async def process_all_queries(self, query_column: str, dataset):
        """Process all queries efficiently with parallel processing and batch I/O."""
        self.qrels_file = os.path.join(self.qrels_file_base, f"{query_column}.json")

        if query_column not in dataset.column_names:
            logging.info(f"Query column '{query_column}' not found in dataset. Skipping.")
            return
        
        # Load existing results
        if os.path.exists(self.qrels_file):
            with open(self.qrels_file, "r") as f:
                existing_qrels = json.load(f)
        else:
            existing_qrels = {}

        # Find queries to process
        queries_to_process = []
        for row in dataset:
            if row[query_column] is not None and row[query_column] not in existing_qrels:
                queries_to_process.append(row[query_column])
        
        if not queries_to_process:
            logging.info(f"No new queries to process for column '{query_column}'")
            return
            
        print(f"🔍 Processing {len(queries_to_process)} queries for '{query_column}' in parallel...")
        
        # Process queries in parallel (BM25 is fast, so we can handle many at once)
        start_time = time.monotonic()
        
        # Process with progress bar
        tasks = [self.process_query(query) for query in queries_to_process]
        results = []
        
        with tqdm(total=len(tasks), desc=f"Querying {query_column}") as pbar:
            # Process in smaller batches to show progress
            batch_size = min(50, len(tasks))  # Process 50 at a time
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch)
                results.extend(batch_results)
                pbar.update(len(batch))
        
        end_time = time.monotonic()
        
        # Batch update results
        for query, retrieved in results:
            if retrieved:  # Only store successful results
                existing_qrels[query] = retrieved
        
        # Single file write at the end
        with open(self.qrels_file, "w") as f:
            json.dump(existing_qrels, f, indent=4)
            
        logging.info(f"Completed {len(queries_to_process)} queries for '{query_column}' in {end_time - start_time:.2f}s")
        logging.info(f"Average query time: {(end_time - start_time) / len(queries_to_process):.3f}s")

    async def __call__(self):
        """
        Main entry point - optimized for speed.
        """
        start_time = time.monotonic()
        logging.info(f"Starting {self.__class__.__name__} retrieval...")
        
        # Prepare corpus once
        await self.prepare_corpus()
        corpus_time = time.monotonic()
        logging.info(f"Corpus preparation took {corpus_time - start_time:.2f}s")

        try:
            # Load dataset only once (major performance improvement!)
            print("📊 Loading dataset...")
            dataset_start = time.monotonic()
            dataset = load_dataset_for_benchmark(os.getenv("DATASET"))
            dataset_time = time.monotonic()
            print(f"✅ Dataset loaded in {dataset_time - dataset_start:.2f}s")
            
            # Process all query columns efficiently
            query_columns = ["query", "rephrase_level_1", "rephrase_level_2", "rephrase_level_3"]
            available_columns = [col for col in query_columns if col in dataset.column_names]
            print(f"📋 Found {len(available_columns)} query columns: {available_columns}")
            
            for query_column in available_columns:
                print(f"\n🎯 Processing '{query_column}' queries...")
                await self.process_all_queries(query_column, dataset)

        except Exception as e:
            logging.error(f"Error during {self.__class__.__name__} retrieval run: {e}", exc_info=True)

        finally:
            total_time = time.monotonic() - start_time
            logging.info(f"{self.__class__.__name__} retrieval completed in {total_time:.2f}s")
            logging.info(f"Processed {len(self.timing_logs)} queries total")
            
            # Calculate and log performance stats
            if self.timing_logs:
                avg_query_time = sum(log["retrieval_duration_sec"] for log in self.timing_logs) / len(self.timing_logs)
                logging.info(f"Average query retrieval time: {avg_query_time:.3f}s")
                successful_queries = sum(1 for log in self.timing_logs if log["status"] == "Success")
                logging.info(f"Success rate: {successful_queries}/{len(self.timing_logs)} ({100*successful_queries/len(self.timing_logs):.1f}%)") 