import os
import asyncio
import logging
import time
import json
from nltk.tokenize import wordpunct_tokenize
from rank_bm25 import BM25Okapi
from data_collection.dataset_loader import load_dataset_for_benchmark
from data_collection.pipelines import TASK
from tqdm import tqdm


class BaseSparseRetrieval():
    """Base class for sparse retrieval baselines (BM25, SPLADE).

    The corpus is the page text straight from the dataset — the same unit the dense text
    pipelines (TEXT-SINGLE / TEXT-MULTI) embed — so the comparison is apples-to-apples.
    Results are written in the same qrels shape as the dense pipelines
    ({query: {"results": {...}, "search_latency": s}}) so the existing eval/aggregation
    pick them up without special-casing.
    """

    def __init__(self, top_k: int = 5, **kwargs):
        self.semaphore = asyncio.Semaphore(120)
        self.hf_token = os.getenv("HF_TOKEN")
        self.qrels_file_base = os.path.join(
            os.getcwd(), f"qrels/{TASK}/{os.getenv('PIPELINE_NAME', 'SPARSE')}/"
        )
        os.makedirs(self.qrels_file_base, exist_ok=True)

        self.top_k = top_k
        self.task = TASK
        self.timing_logs = []

    async def prepare_corpus(self, dataset):
        """Build the corpus from the dataset's page text (one document per image_filename)."""
        corpus = []
        self.documents = []
        seen = set()

        for row in tqdm(dataset, desc="Building corpus"):
            filename = row["image_filename"]
            text = row["text"]
            if filename in seen or text is None:
                continue
            seen.add(filename)
            corpus.append(text)
            self.documents.append({"text": text, "filename": filename})

        self.corpus = corpus
        logging.info(f"Prepared corpus with {len(corpus)} documents")

        print("Building retrieval index...")
        await self._prepare_retrieval_model()

    async def _prepare_retrieval_model(self):
        self.bm25 = BM25Okapi([wordpunct_tokenize(text.casefold()) for text in self.corpus])

    async def search(self, query, top_k=None):
        """Search the page-level corpus and keep the best score per filename."""
        top_k = self.top_k if top_k is None else top_k
        scores = self.bm25.get_scores(wordpunct_tokenize(query.casefold()))
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
        return {
            self.documents[index]["filename"]: float(scores[index])
            for index in ranked
        }

    async def process_query(self, query):
        """Retrieve for a single query, timing the search."""
        try:
            async with self.semaphore:
                start = time.monotonic()
                retrieved = await self.search(query, top_k=self.top_k)
                latency = time.monotonic() - start

            self.timing_logs.append({"query": query, "search_latency": latency})
            return query, retrieved, latency

        except Exception as e:
            logging.error(f"Error during {self.__class__.__name__} retrieval for query {query}: {e}")
            return query, {}, 0.0

    async def process_all_queries(self, query_column: str, dataset):
        """Retrieve for every (deduplicated) query in the column."""
        self.qrels_file = os.path.join(self.qrels_file_base, f"{query_column}.json")

        if query_column not in dataset.column_names:
            logging.info(f"Query column '{query_column}' not found in dataset. Skipping.")
            return

        if os.path.exists(self.qrels_file):
            with open(self.qrels_file, "r") as f:
                existing_qrels = json.load(f)
        else:
            existing_qrels = {}

        # Dedup by query text: multi-positive datasets repeat a query once per gold page.
        queries_to_process = []
        seen_queries = set()
        for row in dataset:
            query = row[query_column]
            if query is not None and query not in existing_qrels and query not in seen_queries:
                seen_queries.add(query)
                queries_to_process.append(query)

        if not queries_to_process:
            logging.info(f"No new queries to process for column '{query_column}'")
            return

        print(f"Processing {len(queries_to_process)} queries for '{query_column}'...")
        results = []
        batch_size = 50
        with tqdm(total=len(queries_to_process), desc=f"Querying {query_column}") as pbar:
            for i in range(0, len(queries_to_process), batch_size):
                batch = [self.process_query(q) for q in queries_to_process[i:i + batch_size]]
                results.extend(await asyncio.gather(*batch))
                pbar.update(len(batch))

        for query, retrieved, latency in results:
            if retrieved:
                existing_qrels[query] = {"results": retrieved, "search_latency": latency}

        with open(self.qrels_file, "w") as f:
            json.dump(existing_qrels, f, indent=4)

        logging.info(f"Wrote {len(existing_qrels)} queries to {self.qrels_file}")

    async def __call__(self):
        start_time = time.monotonic()
        logging.info(f"Starting {self.__class__.__name__} retrieval for {TASK}...")

        print("Loading dataset...")
        dataset = load_dataset_for_benchmark(os.getenv("DATASET"))

        await self.prepare_corpus(dataset)
        await self.process_all_queries("query", dataset)

        total = time.monotonic() - start_time
        if self.timing_logs:
            avg = sum(log["search_latency"] for log in self.timing_logs) / len(self.timing_logs)
            logging.info(f"Avg query search latency: {avg:.4f}s over {len(self.timing_logs)} queries")
        logging.info(f"{self.__class__.__name__} completed in {total:.2f}s")


class BM25Retrieval(BaseSparseRetrieval):
    """First-class BM25 retrieval pipeline."""


async def main():
    await BM25Retrieval()()


if __name__ == "__main__":
    asyncio.run(main())
