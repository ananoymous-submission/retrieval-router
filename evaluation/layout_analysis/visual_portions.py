import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from evaluation.layout_analysis import LayoutAnalyzer


def load_dataset_for_benchmark(dataset_name: str):
    """Load dataset from HuggingFace and concatenate all splits"""
    dataset = load_dataset(dataset_name, token=os.getenv("HF_TOKEN"))

    all_splits = []
    for key in dataset.keys():
        all_splits.append(dataset[key])

    dataset_across_splits = concatenate_datasets(all_splits)
    return dataset_across_splits


class VisualPortionProcessor:
    """Processes datasets to extract visual portions using LayoutAnalyzer"""

    def __init__(self, semaphore_limit: int = 20, batch_size: int = 100):
        """
        Initialize the visual portion processor.

        Args:
            semaphore_limit: Max concurrent layout analysis tasks (default: 5)
            batch_size: Number of documents to process per batch (default: 100)
        """
        self.analyzer = LayoutAnalyzer()
        self.semaphore_limit = semaphore_limit
        self.batch_size = batch_size

    async def process_image_with_semaphore(self, image, semaphore: asyncio.Semaphore) -> float:
        """
        Process a single image with semaphore control.

        Args:
            image: PIL Image from dataset
            semaphore: Semaphore for concurrency control

        Returns:
            Visual portion as float (0.0 to 1.0), or None if image is invalid
        """
        async with semaphore:
            loop = asyncio.get_event_loop()
            try:
                visual_portion = await loop.run_in_executor(
                    None,
                    self.analyzer.process_image,
                    image
                )
                return visual_portion
            except (ValueError, TypeError, IOError) as e:
                return None

    async def process_dataset(self, dataset_name: str, hf_dataset_id: str) -> Dict[str, float]:
        """
        Process all documents in a dataset and save results to JSON.

        Args:
            dataset_name: Name for output file (e.g., 'finreport')
            hf_dataset_id: HuggingFace dataset ID (e.g., 'ananoymous/FinReport')

        Returns:
            Dictionary mapping image_filename to visual_portion
        """
        print(f"\nProcessing dataset: {dataset_name}")

        # Load dataset
        dataset = load_dataset_for_benchmark(hf_dataset_id)
        total_docs = len(dataset)

        print(f"Loaded {total_docs} documents from {hf_dataset_id}")

        # Initialize results and semaphore
        results_dict: Dict[str, float] = {}
        semaphore = asyncio.Semaphore(self.semaphore_limit)

        # Process in batches with progress bar
        batch = []
        skipped_count = 0

        with tqdm(total=total_docs, desc=f"Processing {dataset_name}", unit="docs") as pbar:
            for row in dataset:
                image_filename = row["image_filename"]
                image = row.get("image")

                # Skip rows without images
                if image is None:
                    skipped_count += 1
                    pbar.update(1)
                    continue

                # Create async task
                task = self.process_image_with_semaphore(image, semaphore)
                batch.append((image_filename, task))

                # Process batch when it reaches batch_size
                if len(batch) >= self.batch_size:
                    # Gather all results
                    batch_results = await asyncio.gather(*[t for _, t in batch])

                    # Store results (skip None results from invalid images)
                    for (filename, _), visual_portion in zip(batch, batch_results):
                        if visual_portion is not None:
                            results_dict[filename] = visual_portion
                        else:
                            skipped_count += 1

                    # Update progress
                    pbar.update(len(batch))
                    pbar.set_postfix({"last_file": batch[-1][0], "skipped": skipped_count})

                    # Clear batch
                    batch = []

            # Process remaining documents
            if batch:
                batch_results = await asyncio.gather(*[t for _, t in batch])
                for (filename, _), visual_portion in zip(batch, batch_results):
                    if visual_portion is not None:
                        results_dict[filename] = visual_portion
                    else:
                        skipped_count += 1
                pbar.update(len(batch))

        # Write results to JSON file
        output_file = Path(f"{dataset_name}_visual_portions.json")
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"✓ Saved {len(results_dict)} results to {output_file}")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} documents with missing or invalid images")

        return results_dict

    async def process_all_datasets(self):
        """Process all 6 datasets sequentially"""

        # Dataset configuration
        datasets = [
            # ("finreport", "ananoymous/FinReport"),
            # ("finslides", "ananoymous/FinSlides"),
            # ("finqa", "ananoymous/FinQA"),
            # ("convfinqa", "ananoymous/ConvFinQA"),
            # ("vqaonbd", "ananoymous/VQAonBD"),
            # ("tatdqa", "ananoymous/TATDQA"),
            ("mp-docvqa", "ananoymous/MP-DocVQA"),
            ("mp-sciqag", "ananoymous/SciQAG"),
            ("dude", "ananoymous/DUDE"),
            ("wiki-ss", "ananoymous/Wiki-ss"),
            ("arxivqa", "ananoymous/ArxivQA"),
        ]

        print("=" * 60)
        print("Visual Portion Processing - Starting")
        print("=" * 60)
        print(f"Datasets to process: {len(datasets)}")
        print(f"Semaphore limit: {self.semaphore_limit}")
        print(f"Batch size: {self.batch_size}")
        print("=" * 60)

        total_processed = 0

        # Process each dataset sequentially
        for dataset_name, hf_dataset_id in datasets:
            results = await self.process_dataset(dataset_name, hf_dataset_id)
            total_processed += len(results)

        print("\n" + "=" * 60)
        print("Visual Portion Processing - Complete")
        print("=" * 60)
        print(f"Total documents processed: {total_processed}")
        print(f"Total datasets processed: {len(datasets)}")
        print("=" * 60)


if __name__ == "__main__":
    processor = VisualPortionProcessor()
    asyncio.run(processor.process_all_datasets())
