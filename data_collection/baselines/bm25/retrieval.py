import os
import asyncio
import logging
import time
from datasets import load_dataset

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from data_collection.baselines.base_sparse_retrieval import BaseSparseRetrieval

import nltk
nltk.download('punkt')

class BM25:
    def __init__(self, task):
        self.task = task
        self.preprocessor = None  # Will be set by parent class
        
    async def prepare_corpus(self, corpus, documents):
        """Prepare BM25 index from corpus - this is the only heavy operation."""
        self.documents = documents
        self.corpus = corpus
        
        logging.info(f"Tokenizing {len(corpus)} documents for BM25...")
        start_time = time.time()
        
        # Tokenize all documents (this is the main computational cost)
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        end_time = time.time()
        logging.info(f"BM25 index built in {end_time - start_time:.2f}s")

    def get_top_documents(self, query, top_k=10):
        """Get top documents synchronously - BM25 search is very fast."""
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        scored_docs = [(scores[i], i) for i in range(len(scores))]
        return sorted(scored_docs, key=lambda x: x[0], reverse=True)[:top_k]

    async def search(self, query, top_k=10):
        """Search efficiently - minimal async overhead."""
        top_docs = self.get_top_documents(query, top_k)
        
        results = {}
        for score, doc_idx in top_docs:
            doc = self.documents[doc_idx]
            filename = doc['filename']
            
            # BM25 should return highest score per filename (in case of multiple chunks)
            if filename not in results or score > results[filename]:
                results[filename] = score
        
        return results
    

class BM25Retrieval(BaseSparseRetrieval):
    def __init__(
        self,
        top_k: int = 5,
        **kwargs
    ):
        super().__init__(top_k=top_k, **kwargs)
        
        self.bm25_model = BM25(self.task)
        
    async def _prepare_retrieval_model(self):
        """Prepare the BM25 model with the corpus."""
        await self.bm25_model.prepare_corpus(self.corpus, self.documents)

    async def search(self, query, top_k=None):
        """Search using BM25."""
        if top_k is None:
            top_k = self.top_k
        return await self.bm25_model.search(query, top_k)

async def main():
    """Main function that uses environment variables for configuration."""
    bm25_retrieval = BM25Retrieval()
    await bm25_retrieval()
    
if __name__ == "__main__":
    asyncio.run(main()) 