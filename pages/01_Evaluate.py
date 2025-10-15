import time, pickle, os
import streamlit as st
from app.qa import ExtractiveQA
from app.rerank import Reranker
from app.utils import load_eval_yaml

st.set_page_config(page_title="Mini-RAG • Evaluate", page_icon="✅", layout="wide")

STORE = "store"
INDEX_PKL = os.path.join(STORE, "hybrid.pkl")

@st.cache_resource
def get_models():
    return ExtractiveQA(), Reranker()

def require_index():
    if not os.path.exists(INDEX_PKL):
        st.error("No index found. Go to the main page, upload PDFs, and build the index first.")
        st.stop()
    with open(INDEX_PKL, "rb") as f:
        return pickle.load(f)  # HybridSearcher

st.title("Quick Evaluation")

data = load_eval_yaml()
suite = st.selectbox("Choose eval suite", options=list(data.keys()), format_func=lambda k: data[k].get("doc_hint", k))
run_btn = st.button("Run")

if run_btn and suite:
    hs = require_index()
    qa, rr = get_models()
    pairs = data[suite]["pairs"]

    hits = em = 0
    rows = []
    t0 = time.time()

    for i, item in enumerate(pairs, 1):
        q = item["q"].strip()
        gold = item["gold"]
        cand_idx = hs.hybrid(q, 10, 10, 12)
        candidates = [{"text": hs.corpus[i], "meta": hs.docs[i]} for i,_ in cand_idx]
        top = rr.rerank(q, candidates, top_k=5)
        pred = qa.answer(q, [c["text"] for c in top])["answer"].strip().lower()

        # retrieval hit if any gold token shows in any top chunk
        gold_parts = [g.strip().lower() for g in gold.split("|") if g.strip()]
        hit = any(any(g in c["text"].lower() for g in gold_parts) for c in top)
        exact = any(g in pred for g in gold_parts)

        hits += int(hit); em += int(exact)
        rows.append((q, pred, "✅" if hit else "❌", "✅" if exact else "❌"))

    latency = time.time() - t0
    st.subheader("Results")
    st.write(f"Retrieval hit-rate: **{hits}/{len(pairs)} = {hits/len(pairs):.2f}**")
    st.write(f"Exact-match-ish: **{em}/{len(pairs)} = {em/len(pairs):.2f}**")
    st.caption(f"Latency: {latency:.2f}s")

    for q, pred, hit_flag, em_flag in rows:
        st.markdown(f"- **Q:** {q}\n  \n  **Pred:** {pred}\n  \n  Retrieval: {hit_flag} • EM-ish: {em_flag}")
