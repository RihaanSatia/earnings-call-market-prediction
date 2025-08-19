import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path


class EarningsVectorStore:
    def __init__(self, 
                 persist_directory: str = "data/vector_store",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Store documents and metadata separately
        self.documents = []
        self.metadatas = []
        
        # Try to load existing data
        self._load()
        
        print(f"Vector store initialized with {len(self.documents)} documents")
    
    def _save(self):
        """Save index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, str(self.persist_directory / "index.faiss"))
        
        # Save documents and metadata
        with open(self.persist_directory / "documents.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "metadatas": self.metadatas}, f)
    
    def _load(self):
        """Load index and metadata from disk."""
        index_path = self.persist_directory / "index.faiss"
        data_path = self.persist_directory / "documents.pkl"
        
        if index_path.exists() and data_path.exists():
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load documents and metadata
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadatas = data["metadatas"]
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to vector store."""
        if not chunks:
            return
        
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        
        # Save to disk
        self._save()
        
        print(f"Successfully added {len(chunks)} documents to vector store")
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               company_filter: Optional[str] = None,
               date_filter: Optional[str] = None,
               section_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        search_k = min(len(self.documents), n_results * 3)  # Get more results for filtering
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Format and filter results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.documents):  # Invalid index
                continue
                
            metadata = self.metadatas[idx]
            
            # Apply filters
            if company_filter and metadata.get('company') != company_filter:
                continue
            if date_filter and metadata.get('date') != date_filter:
                continue
            if section_filter and metadata.get('section') != section_filter:
                continue
            
            results.append({
                'text': self.documents[idx],
                'metadata': metadata,
                'similarity_score': float(score),
                'id': idx
            })
            
            if len(results) >= n_results:
                break
        
        return results
    
    def get_companies(self) -> List[str]:
        """Get list of companies in vector store."""
        companies = set()
        for metadata in self.metadatas:
            if 'company' in metadata:
                companies.add(metadata['company'])
        return sorted(list(companies))
    
    def get_dates_for_company(self, company: str) -> List[str]:
        """Get earnings dates for a specific company."""
        dates = set()
        for metadata in self.metadatas:
            if metadata.get('company') == company and 'date' in metadata:
                dates.add(metadata['date'])
        return sorted(list(dates))
    
    def delete_all(self) -> None:
        """Delete all documents from vector store."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []
        self.metadatas = []
        self._save()
        print("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        companies = self.get_companies()
        
        return {
            "total_documents": len(self.documents),
            "total_companies": len(companies),
            "companies": companies,
            "embedding_dimension": self.embedding_dim
        }