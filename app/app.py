import os, pickle, logging, time
import streamlit as st
from app.ingest import read_pdf, chunk_text, to_documents
from app.search import HybridSearcher
from app.rerank import Reranker
from app.qa import ExtractiveQA

st.set_page_config(page_title="Mini-RAG", page_icon="🔎", layout="wide")
logging.basicConfig(level=logging.INFO)

STORE = "store"
os.makedirs(STORE, exist_ok=True)

@st.cache_resource
def get_models():
    return ExtractiveQA(), Reranker()

def build_or_load(docs):
    idx_pkl = os.path.join(STORE, "hybrid.pkl")
    if os.path.exists(idx_pkl):
        with open(idx_pkl, "rb") as f:
            return pickle.load(f)
    hs = HybridSearcher(docs)
    with open(idx_pkl, "wb") as f:
        pickle.dump(hs, f)
    return hs

st.title("Mini-RAG: Hybrid Search + Rerank + QA")

uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded:
    all_docs = []
    for up in uploaded:
        path = os.path.join("data", up.name)
        os.makedirs("data", exist_ok=True)
        with open(path, "wb") as f: f.write(up.read())
        text = read_pdf(path)
        chunks = chunk_text(text)
        all_docs += to_documents(chunks, path)

    with st.spinner("Indexing..."):
        hs = build_or_load(all_docs)
    st.success(f"Indexed {len(hs.corpus)} chunks.")

    qa, rr = get_models()

    q = st.text_input("Ask a question about your PDFs")
    if q:
        t0 = time.time()
        cand_idx = hs.hybrid(q, 10, 10, 12)
        candidates = [{"text": hs.corpus[i], "meta": hs.docs[i]} for i, _ in cand_idx]
        top = rr.rerank(q, candidates, top_k=5)
        contexts = [c["text"] for c in top]
        out = qa.answer(q, contexts)
        latency = time.time() - t0

        st.subheader("Answer")
        st.write(out["answer"])
        st.caption(f"confidence ~ {out['score']:.3f} | latency {latency:.2f}s")

        st.subheader("Sources")
        for c in top:
            st.markdown(f"- **{c['meta']['source']}**: {c['text'][:300]}...")
else:
    st.info("Upload one or more PDFs to begin.")
