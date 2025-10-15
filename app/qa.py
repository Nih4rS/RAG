from __future__ import annotations
from typing import Dict, List
from transformers import pipeline

class ExtractiveQA:
    def __init__(self, model_name: str = "deepset/roberta-base-squad2"):
        self.pipe = pipeline("question-answering", model=model_name, tokenizer=model_name)

    def answer(self, question: str, contexts: List[str], max_ctx_chars: int = 3500) -> Dict:
        ctx = " ".join(contexts)
        ctx = ctx[:max_ctx_chars]
        out = self.pipe({"question": question, "context": ctx})
        return {"answer": out.get("answer", ""), "score": float(out.get("score", 0.0))}
