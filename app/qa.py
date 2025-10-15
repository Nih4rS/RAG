"""Extractive QA pipeline."""

from typing import List, Dict, Optional
from transformers import pipeline


class ExtractiveQA:
    """Extractive question answering system."""
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initialize the extractive QA pipeline.
        
        Args:
            model_name: Name of the QA model to use
        """
        self.qa_pipeline = pipeline("question-answering", model=model_name)
    
    def answer_question(
        self, 
        question: str, 
        contexts: List[str], 
        top_k: int = 1
    ) -> List[Dict]:
        """
        Answer a question given context passages.
        
        Args:
            question: The question to answer
            contexts: List of context passages
            top_k: Number of answers to return
        
        Returns:
            List of answers with scores and context
        """
        if not contexts:
            return []
        
        answers = []
        
        for i, context in enumerate(contexts):
            try:
                result = self.qa_pipeline(
                    question=question,
                    context=context
                )
                
                answers.append({
                    'answer': result['answer'],
                    'score': result['score'],
                    'context': context,
                    'context_index': i,
                    'start': result['start'],
                    'end': result['end']
                })
            except Exception as e:
                # Skip contexts that cause errors
                continue
        
        # Sort by score
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)
        
        return answers[:top_k]
    
    def answer_with_search_results(
        self, 
        question: str, 
        search_results: List[Dict], 
        top_k: int = 1
    ) -> List[Dict]:
        """
        Answer a question using search results.
        
        Args:
            question: The question to answer
            search_results: List of search results with 'chunk' field
            top_k: Number of answers to return
        
        Returns:
            List of answers with metadata
        """
        contexts = [result['chunk'] for result in search_results]
        answers = self.answer_question(question, contexts, top_k)
        
        # Add metadata from search results
        for answer in answers:
            context_idx = answer['context_index']
            if context_idx < len(search_results):
                answer['metadata'] = search_results[context_idx].get('metadata', {})
        
        return answers
