import re
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from datetime import datetime
import tiktoken


class TranscriptProcessor:
    def __init__(self, max_tokens: int = 1500, overlap_tokens: int = 300):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 tokenizer
    
    def clean_transcript(self, text: str) -> str:
        """Clean and normalize transcript text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common transcript artifacts
        text = re.sub(r'Thomson Reuters.*?Event Transcript', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\*+\s*\w+.*?\*+', '', text)  # Remove speaker markers
        text = re.sub(r'CORPORATE PARTICIPANTS|CONFERENCE CALL PARTICIPANTS', '', text, flags=re.IGNORECASE)
        
        # Clean up formatting
        text = text.replace('\n\n', '\n').strip()
        
        return text
    
    def extract_sections(self, transcript: str) -> Dict[str, str]:
        """Extract different sections of the earnings call."""
        sections = {
            'presentation': '',
            'qa': '',
            'full': transcript
        }
        
        # Try to split presentation vs Q&A
        qa_markers = [
            'questions and answers',
            'question-and-answer',
            'q&a session',
            'operator.*questions'
        ]
        
        for marker in qa_markers:
            pattern = rf'({marker}.*?)$'
            match = re.search(pattern, transcript, re.IGNORECASE | re.DOTALL)
            if match:
                qa_start = match.start()
                sections['presentation'] = transcript[:qa_start].strip()
                sections['qa'] = transcript[qa_start:].strip()
                break
        
        if not sections['presentation']:
            sections['presentation'] = transcript
        
        return sections
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks using efficient batch tokenization."""
        # Tokenize the full text once
        tokens = self.tokenizer.encode(text)
        
        # If text is small enough, return as single chunk
        if len(tokens) <= self.max_tokens:
            return [{
                'text': text,
                'metadata': metadata,
                'chunk_id': 0
            }]
        
        # Calculate all chunk boundaries at once
        chunk_starts = []
        start_token = 0
        while start_token < len(tokens):
            chunk_starts.append(start_token)
            start_token += (self.max_tokens - self.overlap_tokens)
        
        # Extract all chunk token sequences
        chunk_token_lists = []
        for i, start in enumerate(chunk_starts):
            end = min(start + self.max_tokens, len(tokens))
            chunk_token_lists.append(tokens[start:end])
        
        # Batch decode all chunks at once
        chunk_texts = self.tokenizer.decode_batch(chunk_token_lists)
        
        # Build final chunk objects
        chunks = []
        for i, (chunk_text, chunk_tokens) in enumerate(zip(chunk_texts, chunk_token_lists)):
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_id': i,
                        'token_count': len(chunk_tokens)
                    },
                    'chunk_id': i
                })
        
        return chunks
    
    def process_transcript(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single transcript into chunks."""
        transcript = transcript_data['transcript']
        company = transcript_data['company']
        date = transcript_data['date']
        
        # Clean transcript
        cleaned_transcript = self.clean_transcript(transcript)
        
        # Extract sections
        sections = self.extract_sections(cleaned_transcript)
        
        all_chunks = []
        
        # Process each section
        for section_name, section_text in sections.items():
            if section_text and section_name != 'full':
                # Convert date to timestamp for range filtering
                try:
                    date_timestamp = datetime.strptime(date, "%Y-%m-%d").timestamp()
                except:
                    date_timestamp = 0  # fallback
                
                base_metadata = {
                    'company': company,
                    'date': date,
                    'date_timestamp': date_timestamp,
                    'section': section_name,
                    'doc_type': 'earnings_transcript'
                }
                
                section_chunks = self.chunk_text(section_text, base_metadata)
                all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def process_dataset(self, transcripts_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process entire dataset of transcripts."""
        all_chunks = []
        
        for _, row in transcripts_df.iterrows():
            transcript_data = {
                'transcript': row['transcript'],
                'company': row['company'],
                'date': row['date']
            }
            
            chunks = self.process_transcript(transcript_data)
            all_chunks.extend(chunks)
            
            print(f"Processed {row['company']} {row['date']}: {len(chunks)} chunks")
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks