import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid


class EarningsVectorStore:
    def __init__(self, 
                 persist_directory: str = "data/vector_store",
                 collection_name: str = "earnings_transcripts"):
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        doc_count = self.collection.count()
        print(f"Vector store initialized with {doc_count} documents")
    
    
    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 500) -> None:
        """Add document chunks to vector store in batches."""
        if not chunks:
            return
        
        total_chunks = len(chunks)
        print(f"Adding {total_chunks} documents to vector store in batches of {batch_size}...")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            # Prepare data for Chroma
            ids = [str(uuid.uuid4()) for _ in batch]
            documents = [chunk['text'] for chunk in batch]
            metadatas = [chunk['metadata'] for chunk in batch]
            
            # Add to Chroma collection
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"  Batch {batch_num} completed successfully")
            except Exception as e:
                print(f"  Error in batch {batch_num}: {e}")
                raise
        
        print(f"Successfully added all {total_chunks} documents to vector store")
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               company_filter: Optional[str] = None,
               date_filter: Optional[str] = None,
               section_filter: Optional[str] = None,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for relevant document chunks with metadata filtering."""
        
        # Use new filters parameter if provided, otherwise fall back to individual filters
        if filters:
            where_clause = filters
        else:
            # Build where clause for filtering (backwards compatibility)
            where_clause = {}
            if company_filter:
                where_clause["company"] = company_filter
            if date_filter:
                where_clause["date"] = date_filter
            if section_filter:
                where_clause["section"] = section_filter
        
        # Search with Chroma
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'id': results['ids'][0][i]
            })
        
        return formatted_results
    
    def get_companies(self) -> List[str]:
        """Get list of companies in vector store."""
        # Get all documents and extract unique companies
        all_docs = self.collection.get()
        companies = set()
        for metadata in all_docs['metadatas']:
            if 'company' in metadata:
                companies.add(metadata['company'])
        return sorted(list(companies))
    
    def get_dates_for_company(self, company: str) -> List[str]:
        """Get earnings dates for a specific company."""
        # Query documents for specific company
        company_docs = self.collection.get(
            where={"company": company}
        )
        dates = set()
        for metadata in company_docs['metadatas']:
            if 'date' in metadata:
                dates.add(metadata['date'])
        return sorted(list(dates))
    
    def delete_all(self) -> None:
        """Delete all documents from vector store."""
        # Delete the collection and recreate it
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        companies = self.get_companies()
        doc_count = self.collection.count()
        
        return {
            "total_documents": doc_count,
            "total_companies": len(companies),
            "companies": companies,
            "collection_name": self.collection_name
        }