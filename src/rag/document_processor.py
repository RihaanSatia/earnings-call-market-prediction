import re
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd


class TranscriptProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
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
        """Split text into overlapping chunks with metadata."""
        if len(text) <= self.chunk_size:
            return [{
                'text': text,
                'metadata': metadata,
                'chunk_id': 0
            }]
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_id': chunk_id,
                        'start_pos': start,
                        'end_pos': end
                    },
                    'chunk_id': chunk_id
                })
                chunk_id += 1
            
            start = end - self.chunk_overlap
        
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
                base_metadata = {
                    'company': company,
                    'date': date,
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