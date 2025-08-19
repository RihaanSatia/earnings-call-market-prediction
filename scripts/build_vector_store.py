#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.document_processor import TranscriptProcessor
from rag.vector_store import EarningsVectorStore


def main():
    print("Building vector store from earnings transcripts...")
    
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    transcripts_df = pd.read_csv(data_dir / "earnings_calls_transcripts.csv")
    
    print(f"Loaded {len(transcripts_df)} transcripts")
    
    processor = TranscriptProcessor(chunk_size=1000, chunk_overlap=200)
    vector_store = EarningsVectorStore()
    
    vector_store.delete_all()
    
    print("Processing transcripts...")
    chunks = processor.process_dataset(transcripts_df)
    
    print("Adding to vector store...")
    vector_store.add_documents(chunks)
    
    stats = vector_store.get_stats()
    print(f"\nVector store built successfully!")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Companies: {stats['companies']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")


if __name__ == "__main__":
    main()