"""
Smart search parser for earnings call queries
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import openai

from utils.prompts import SEARCH_PARSING_PROMPT


@dataclass
class SearchParams:
    companies: List[str]
    date_start: Optional[str]
    date_end: Optional[str]
    search_terms: List[str]


class SmartSearchParser:
    def __init__(self):
        self.client = openai.OpenAI()
        
    def parse_query(self, query: str) -> SearchParams:
        """Parse user query using LLM to extract search parameters"""
        return self._llm_parsing(query)
    
    def _llm_parsing(self, query: str) -> SearchParams:
        """Use LLM to parse queries"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": SEARCH_PARSING_PROMPT.format(query=query)}
                ],
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return SearchParams(
                companies=result.get('companies', []),
                date_start=result.get('date_range', {}).get('start'),
                date_end=result.get('date_range', {}).get('end'),
                search_terms=result.get('search_terms', [])
            )
            
        except Exception as e:
            print(f"LLM parsing failed: {e}")
            # Fallback to using original query as search terms
            return SearchParams(
                companies=[],
                date_start=None,
                date_end=None,
                search_terms=[query]
            )
    
    def _date_to_timestamp(self, date_str: str) -> float:
        """Convert date string to timestamp for Chroma filtering"""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.timestamp()
        except Exception as e:
            print(f"Error converting date '{date_str}': {e}")
            return None
    
    def create_filters(self, params: SearchParams) -> Dict[str, Any]:
        """Convert search parameters to Chroma where clause filters"""
        conditions = []
        
        # Add company filter
        if params.companies:
            conditions.append({"company": params.companies[0]})
        
        # Add date filter with timestamp conversion
        if params.date_start and params.date_end:
            start_ts = self._date_to_timestamp(params.date_start)
            end_ts = self._date_to_timestamp(params.date_end)
            if start_ts and end_ts:
                conditions.append({"date_timestamp": {"$gte": start_ts}})
                conditions.append({"date_timestamp": {"$lte": end_ts}})
        elif params.date_start:
            start_ts = self._date_to_timestamp(params.date_start)
            if start_ts:
                conditions.append({"date_timestamp": {"$gte": start_ts}})
        elif params.date_end:
            end_ts = self._date_to_timestamp(params.date_end)
            if end_ts:
                conditions.append({"date_timestamp": {"$lte": end_ts}})
        
        # Return proper Chroma filter format
        if len(conditions) == 0:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
    
    def enhance_query(self, original_query: str, params: SearchParams) -> str:
        """Enhance the original query with extracted search terms"""
        if params.search_terms:
            return ' '.join([original_query] + params.search_terms)
        return original_query