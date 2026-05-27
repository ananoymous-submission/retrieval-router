# RetrievalRouter: Joint Modality and Architecture Selection for Document Retrieval

[![SIGIR '26](https://img.shields.io/badge/EMNLP-2026-blue)](https://2026.emnlp.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<p align="center">
  <a href="#datasets">Benchmarks</a> •
  <a href="#indexing">Indexing</a> •
  <a href="#retrieval">Retrieval</a> •
  <a href="#reranking-two-stage">Reranking</a> •
  <a href="#pre-trained-weights">Pre-trained Weights</a> •
  <a href="#step-1-generate-oracle-labels">Training</a> •
  <a href="#step-3-inference">Inference</a><br>
  <a href="https://huggingface.co/collections/ananoymous/RetrievalRouter">Model & Datasets</a>
</p>

**RetrievalRouter**  is the first system to route jointly across these two axes for a single underlying corpus. By directing each query to the cheapest pipeline that can answer it, the router reserves expensive configurations for queries that genuinely require them. The resulting system attains the accuracy of the strongest static pipeline while incurring an average latency near the cheapest.


 RetrievalRouter dominates every static configuration on the accuracy--latency frontier, achieving 2.7\% higher nDCG@5 than the strongest static baseline at roughly $10\times$ lower latency, and exceeds the deployment-standard reranking baseline by 4.0\% nDCG@5 with a $1.4\times$ speedup.
 

<div align="center">
  <img src="assets/pareto.png" alt="Pareto Frontier" width="600"/>
</div>

| Pipeline | nDCG@5 | Latency (s) | Storage (Index) |
| :--- | :---: | :---: | :---: |
| Text-Dense (TD) | 0.49 | 0.46 | ~3 GB |
| Multimodal-Late (ML) | 0.74 | 8.28 | ~39 GB |
| **RetrievalRouter ($\lambda=0.0$)** | **0.76** | **0.80** | *Hybrid* |
| **RetrievalRouter ($\lambda=0.7$)** | 0.68 | 0.40 | *Hybrid* |

---

## Repository Structure

```
retrieval-router/

├── data_collection/
│   ├── embedders/              # Embedding model implementations
│   │   ├── base_embedder.py    # Abstract base class + factory
│   │   ├── colbert.py          # GTE-ModernColBERT (text late-interaction)
│   │   ├── linq.py             # Linq-Embed-Mistral (text dense)
│   │   ├── voyage.py           # Voyage AI embeddings (text & multimodal)
│   │   ├── colqwen.py          # ColQwen2.5 (multimodal late-interaction)
│   │   └── biqwen.py           # BiQwen2.5 (multimodal dense)
│   ├── pipelines/              # LangGraph-based workflows
│   │   ├── indexing.py         # Document indexing pipeline
│   │   ├── retrieval.py        # Single-stage retrieval
│   │   └── retrieval_rerank.py # Two-stage retrieval + reranking
│   ├── vector_databases/       # Vector store abstraction
│   │   ├── base_database.py    # Abstract base + factory
│   │   └── qdrant_manager.py   # Qdrant implementation
│   └── dataset_loader.py       # HuggingFace dataset loading
│
├── scripts/                    # Shell scripts for running pipelines
│   ├── indexing/
│   │   ├── text/               # Text indexing (dense.sh, multi.sh)
│   │   └── multimodal/         # Multimodal indexing (dense.sh, multi.sh)
│   └── retrieval/
│       ├── text/               # Text retrieval (dense.sh, multi.sh, two_stage.sh)
│       └── multimodal/         # Multimodal retrieval (dense.sh, multi.sh, two_stage.sh)
│
├── train/                      # Router training
│   ├── config.py               # Hyperparameters and model config
│   ├── model.py                # QwenForSoftClassification
│   ├── loss.py                 # Weighted KL divergence loss
│   ├── prepare_data.py         # Oracle label generation
│   ├── train.py                # Training loop
│   └── inference.py            # Model inference
│
├── evaluation/                 # Evaluation scripts
│   └── overall_metrics.py      # Compute nDCG, MRR, Recall, Latency
│
└── requirements/               # Dependencies
    ├── base_pipelines_requirements.txt
    ├── training_requirements.txt
    └── layout_analysis_requirements.txt
```

## Datasets

We evaluate on 11 benchmarks spanning financial, scientific, and open domains. You can access all of our datasets in [here](https://huggingface.co/ananoymous/).

| Benchmark | Dataset | # Queries | # Documents | Avg. Tokens |
|-----------|---------|-----------|-------------|-------------|
| REAL-MM-RAG | FinReport | 853 | 2,687 | 1,053 |
| REAL-MM-RAG | FinSlides | 1,048 | 2,280 | 275 |
| T2-RAGBench | FinQA | 6,232 | 2,789 | 965 |
| T2-RAGBench | ConvFinQA | 3,431 | 1,806 | 966 |
| T2-RAGBench | VQAnBD | 9,772 | 1,787 | 780 |
| T2-RAGBench | TAT-DQA | 27,127 | 2,758 | 852 |
| MMDocRAG | ArxivQA | 9,034 | 4,749 | 1,110 |
| MMDocRAG | Wiki-SS | 14,968 | 12,752 | 777 |
| MMDocRAG | MP-DocVQA | 5,581 | 2,350 | 388 |
| MMDocRAG | SciQAG | 4,496 | 2,595 | 1,196 |
| MMDocRAG | DUDE | 2,561 | 2,073 | 516 |

**Total: 80,000+ queries**

---

# Part 1: Base Retrieval Pipelines

This section covers indexing documents and running retrieval/reranking with the base pipelines.

## Installation (Base Pipelines)

```bash
# Clone the repository
git clone https://github.com/anonymous/sigir26.git
cd sigir26

# Create virtual environment
conda create -n RetrievalRouter python=3.10
conda activate RetrievalRouter

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install base pipeline dependencies
pip install -r requirements/base_pipelines_requirements.txt
```

**System Requirements:**
- Python >= 3.10
- CUDA >= 12.0
- **Poppler:** Required for PDF rendering in multimodal pipelines.
    - Ubuntu: `sudo apt-get install poppler-utils`
    - Mac: `brew install poppler`

## Pipeline Strategies

| Strategy | Embedder | Description |
|----------|----------|-------------|
| TEXT-DENSE | Linq-Embed-Mistral | Text-Dense retrieval |
| TEXT-LATE | GTE-ModernColBERT | Text-Late interaction retrieval |
| MULTIMODAL-DENSE | BiQwen2.5 | Multimodal-Dense retrieval |
| MULTIMODAL-LATE | ColQwen2.5 | Multimodal-Late interaction retrieval |
| TEXT-RERANK | Linq + ColBERT | Text-Dense → Text-Late reranking |
| MULTIMODAL-RERANK | BiQwen + ColQwen | Multimodal-Dense → Multimodal-Late reranking |

## Configuration

Configure your pipeline in `.env`:

```bash
# Pipeline selection
PIPELINE_NAME="MULTIMODAL-SINGLE"  # Options: TEXT-SINGLE, TEXT-MULTI, MULTIMODAL-SINGLE, MULTIMODAL-MULTI, TWO-STAGE

# Two-stage mode (only if PIPELINE_NAME=TWO-STAGE)
TWO_STAGE_MODE="MULTIMODAL"  # Options: TEXT, MULTIMODAL

# Vector database
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY="your-api-key"
COLLECTION_NAME="your-collection"
```

## Indexing

Index documents into the vector database:

```bash
python -m data_collection.pipelines.indexing
```

This will:
1. Load documents from a configured dataset
2. Embed documents using the selected embedder
3. Upsert embeddings into Qdrant with metadata

## Retrieval

Run single-stage retrieval:

```bash
python -m data_collection.pipelines.retrieval
```

## Reranking (Two-Stage)

The two-stage pipeline performs dense retrieval followed by late-interaction reranking:

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                    │
│  ┌──────────────────────────────┐       ┌──────────────────────────────┐           │
│  │   Stage 1: Dense Retrieval   │       │   Stage 2: Late Reranking    │           │
│  └──────────────────────────────┘       └──────────────────────────────┘           │
│                                                                                    │
│  Query → Dense Embedder → Vector Search → Candidates → Late Embedder → Rerank      │
│                              (top-100)                                  (top-5)    │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Multimodal Reranking

```bash
python -m data_collection.pipelines.retrieval_rerank --mode multimodal
```

This uses:
- **Stage 1**: BiQwen2.5 (Multimodal-Dense) → retrieves top-100 candidates
- **Stage 2**: ColQwen2.5 (Multimodal-Late) → reranks to top-5

### Text Reranking

```bash
python -m data_collection.pipelines.retrieval_rerank --mode text
```

This uses:
- **Stage 1**: Linq-Embed-Mistral (Text-Dense) → retrieves top-100 candidates
- **Stage 2**: GTE-ModernColBERT (Text-Late) → reranks to top-5

### Reranking Options

```bash
python -m data_collection.pipelines.retrieval_rerank \
    --mode multimodal \
    --dense-limit 100 \
    --rerank-limit 5 \
    --filter
```

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Pipeline mode: `multimodal` or `text` | `multimodal` |
| `--dense-limit` | Number of candidates from Stage 1 | `100` |
| `--rerank-limit` | Number of final results after Stage 2 | `5` |
| `--filter` | Filter by document name from query metadata | `False` |

## Index Storage

| Pipeline | Index Size |
|----------|------------|
| Text-Dense | ~3 GB |
| Text-Late | ~8 GB |
| Multimodal-Dense | ~3 GB |
| Multimodal-Late | ~39 GB |

---

# Part 2: Router Training

This section covers training the RetrievalRouter query router.

## Installation (Training)

```bash
# Install training dependencies (in addition to base)
pip install -r requirements/training_requirements.txt
```

## Pre-trained Weights

Pre-trained router weights are available on HuggingFace:

| Model | Description | Link |
|-------|-------------|------|
| RetrievalRouter-Base | Router trained with λ=0.1 | [HuggingFace](https://huggingface.co/ananoymous/RetrievalRouter) |

## Step 1: Generate Oracle Labels

Run all pipelines on training data to collect nDCG and latency metrics:

```bash
python -m train.prepare_data
```

This creates reward labels using:
```
r(q, i) = (1 - λ) · nDCG(q, i) + λ · (1 - NormalizedLatency(q, i))
```

Where:
- `λ ∈ [0, 1]`: Trade-off parameter (0 = accuracy-focused, 1 = latency-focused)
- `nDCG(q, i)`: nDCG@5 score for pipeline i on query q
- `NormalizedLatency(q, i)`: Min-max normalized latency per query

## Step 2: Train the Router

```bash
python -m train.train
```

### Router Architecture
- **Encoder**: Qwen3-0.6B (frozen)
- **LoRA**: rank=16, alpha=32
- **Targets**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Classification Head**: Linear layer (hidden_dim → 4 strategies)

### Training Hyperparameters
- **Epochs**: 2
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Scheduler**: Cosine
- **Precision**: bfloat16

## Step 3: Inference

```bash
python -m train.inference --input queries.json --output predictions.xlsx
```

Or use programmatically:

```python
from train.inference import load_model

model = load_model("ananoymous/RetrievalRouter")
prediction = model.predict("What is the revenue for Q3 2024?")
```

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA H100 80GB (full experiments) |
| GPU (inference only) | Standard GPU with 24GB VRAM |
| VRAM | ~40GB for Multimodal-Late index |
| Storage | ~53GB for all vector indices |
| Router Overhead | ~15ms per query |

---

## Citation

```bibtex
@inproceedings{RetrievalRouter2026,
  title={RetrievalRouter: Adaptive Query Routing for Multimodal RAG},
  author={Anonymous Author(s)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds on several open-source projects:
- [ColPali](https://github.com/illuin-tech/colpali) for multimodal late-interaction retrieval
- [PyLate](https://github.com/lightonai/pylate) for ColBERT implementation
- [LangGraph](https://github.com/langchain-ai/langgraph) for pipeline orchestration
- [Qdrant](https://github.com/qdrant/qdrant) for vector storage
