from typing import List, Dict, Any, Optional
from .vector_store import EarningsVectorStore
from .smart_search import SmartSearchParser
from utils.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT
import openai
import os


class EarningsQASystem:
    def __init__(self, 
                 vector_store: Optional[EarningsVectorStore] = None,
                 use_openai: bool = False,
                 openai_model: str = "gpt-3.5-turbo"):
        
        self.vector_store = vector_store or EarningsVectorStore()
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.search_parser = SmartSearchParser()
        
        if use_openai:
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def retrieve_context(self, 
                        question: str, 
                        n_results: int = 5,
                        company_filter: Optional[str] = None,
                        date_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a question."""
        return self.vector_store.search(
            query=question,
            n_results=n_results,
            company_filter=company_filter,
            date_filter=date_filter
        )
    
    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context string."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            company = metadata.get('company', 'Unknown')
            date = metadata.get('date', 'Unknown')
            section = metadata.get('section', 'transcript')
            
            context_parts.append(
                f"[Source {i}] {company} earnings call ({date}, {section}):\n"
                f"{result['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_answer_local(self, question: str, context: str) -> str:
        """Generate answer using local processing (no LLM)."""
        if not context.strip():
            return "I couldn't find relevant information in the earnings call transcripts to answer your question."
        
        return f"Based on the earnings call transcripts, here's the relevant information I found. Please refer to the source documents below for specific details."
    
    def generate_answer_openai(self, question: str, context: str) -> str:
        """Generate answer using OpenAI."""
        prompt = QA_USER_PROMPT.format(question=question, context=context)
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def ask(self, 
            question: str,
            n_results: int = 5) -> Dict[str, Any]:
        """Main QA interface with smart search parsing."""
        
        # Parse query using smart search
        search_params = self.search_parser.parse_query(question)
        enhanced_question = self.search_parser.enhance_query(question, search_params)
        
        # Create filters from parsed parameters
        filters = self.search_parser.create_filters(search_params)
        # Use filters for search - pass the complete filters dict to vector store
        search_results = self.vector_store.search(
            query=enhanced_question,
            n_results=n_results,
            filters=filters
        )
        
        print(f"Found {len(search_results)} chunks for query: {enhanced_question}")
        
        if not search_results:
            return {
                "question": question,
                "answer": "I couldn't find relevant information in the earnings call transcripts for your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        context = self.format_context(search_results)
        
        if self.use_openai and os.getenv("OPENAI_API_KEY"):
            answer = self.generate_answer_openai(question, context)
        else:
            answer = self.generate_answer_local(question, context)
        
        avg_similarity = sum(r['similarity_score'] for r in search_results) / len(search_results)
        
        return {
            "question": question,
            "answer": answer,
            "sources": search_results,
            "confidence": float(avg_similarity)
        }
    
    def get_available_companies(self) -> List[str]:
        """Get list of available companies."""
        return self.vector_store.get_companies()
    
    def get_dates_for_company(self, company: str) -> List[str]:
        """Get available dates for a company."""
        return self.vector_store.get_dates_for_company(company)