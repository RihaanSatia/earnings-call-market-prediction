"""
Essential prompts for RAG-based question answering
"""

QA_SYSTEM_PROMPT = """You are a financial analyst assistant. Answer questions based solely on the provided earnings call transcript excerpts.

Guidelines:
- Answer based only on the provided context
- If the context doesn't contain relevant information, say so clearly
- Cite specific companies and dates when possible
- Be concise but comprehensive
- Focus on factual information from the transcripts"""

QA_USER_PROMPT = """Question: {question}

Earnings Call Context:
{context}

Answer:"""

SEARCH_PARSING_PROMPT = """Extract search parameters from this user query about earnings calls.

Query: "{query}"

Available companies (use ONLY these ticker symbols):
- AAPL (Apple)
- MSFT (Microsoft)  
- GOOGL (Google/Alphabet)
- AMZN (Amazon)
- META (Meta/Facebook)
- TSLA (Tesla)
- NVDA (Nvidia)
- NFLX (Netflix)
- CRM (Salesforce)
- CSCO (Cisco)
- AMD (AMD)
- MU (Micron)
- ASML (ASML)

Extract and return in JSON format:
{{
    "companies": ["list of ticker symbols if mentioned, e.g. AAPL, MSFT"],
    "date_range": {{"start": "YYYY-MM-DD or null", "end": "YYYY-MM-DD or null"}},
    "topics": ["list of main topics/themes to search for"],
    "search_terms": ["key terms for semantic search"]
}}

Rules:
- For companies: ONLY return ticker symbols from the list above. If "Apple" is mentioned, return "AAPL"
- If no specific companies mentioned or company not in list, return empty list
- For dates: Companies typically report quarterly earnings 4-6 weeks AFTER quarter end
  - Q1 (Jan-Mar) earnings are usually reported in April-May
  - Q2 (Apr-Jun) earnings are usually reported in July-August  
  - Q3 (Jul-Sep) earnings are usually reported in October-November
  - Q4 (Oct-Dec) earnings are usually reported in January-February
- If no dates mentioned, return null for both start and end
- Extract 2-5 most relevant search terms focusing on financial topics
- Common topics: earnings, revenue, guidance, competition, risks, growth, margins

Return only valid JSON, no explanation."""