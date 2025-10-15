from __future__ import annotations
from typing import List, Tuple
from app.search import HybridSearcher
from app.rerank import Reranker
from app.qa import ExtractiveQA

def run_eval(hs: HybridSearcher, qa: ExtractiveQA, rr: Reranker, qa_pairs: List[Tuple[str, str]]) -> None:
    hits, em = 0, 0
    for q, gold in qa_pairs:
        cand_idx = hs.hybrid(q, 10, 10, 12)
        candidates = [{"text": hs.corpus[i], "meta": hs.docs[i]} for i,_ in cand_idx]
        top = rr.rerank(q, candidates, top_k=5)
        pred = qa.answer(q, [c["text"] for c in top])["answer"].strip().lower()
        if any(g in c["text"].lower() for c in top for g in gold.split("|")): hits += 1
        if any(g.strip() in pred for g in gold.split("|")): em += 1
    n = len(qa_pairs)
    print(f"Retrieval hit-rate: {hits}/{n} = {hits/n:.2f}")
    print(f"Exact-match-ish:   {em}/{n} = {em/n:.2f}")
