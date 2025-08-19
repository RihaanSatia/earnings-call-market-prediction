#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from rag.qa_system import EarningsQASystem


def main():
    print("Testing QA System...")
    
    qa_system = EarningsQASystem(use_openai= True)
    
    print(f"Available companies: {qa_system.get_available_companies()}")
    
    test_questions = [
        "What did Apple say about iPhone sales?",
        "How did NVIDIA perform in Q4?",
        "What are the revenue growth prospects for AMD?",
        "Tell me about Microsoft's cloud business"
    ]
    
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print('='*50)
        
        result = qa_system.ask(question, n_results=3)
        
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Sources found: {len(result['sources'])}")
        
        for i, source in enumerate(result['sources'], 1):
            metadata = source['metadata']
            print(f"\nSource {i}: {metadata['company']} ({metadata['date']}) - Score: {source['similarity_score']:.3f}")
            print(f"Text preview: {source['text'][:200]}...")


if __name__ == "__main__":
    main()