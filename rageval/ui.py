"""Optional Streamlit UI over the API — a ~40-line front end so you can *see* RAG.

This is intentionally thin: it does NOT re-implement the pipeline. It just POSTs your
question to the running FastAPI service and renders the answer, the cited sources, and
the eval verdict. That separation (UI talks to API, API owns the logic) is the same
shape you'd use in production.

Run it with the API already serving on :8000:

    pip install -e ".[ui]"
    uvicorn rageval.api:app          # terminal 1
    streamlit run rageval/ui.py      # terminal 2
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("RAGEVAL_API_URL", "http://localhost:8000")

st.set_page_config(page_title="rag-eval-demo", page_icon=":books:")
st.title("RAG + Eval demo")
st.caption("Ask a question about the sample 'Lumen Notes' docs. "
           "The answer is generated from retrieved context and then judged for faithfulness.")

question = st.text_input("Your question", value="How do I reset my password?")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving, generating, and evaluating..."):
        try:
            resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001 - surface any error to the user
            st.error(f"Request failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Sources")
    st.write(", ".join(data["sources"]) or "—")

    st.subheader("Eval verdict")
    ev = data["eval"]
    # A green/red banner driven by the gate makes the eval result the headline.
    if ev["overall_pass"]:
        st.success("PASS — answer judged grounded and relevant")
    else:
        st.error("FAIL — answer flagged by the judge")
    col1, col2 = st.columns(2)
    col1.metric("Faithfulness", f'{ev["faithfulness"]["score"]}/5', ev["faithfulness"]["severity"])
    col2.metric("Answer relevance", f'{ev["answer_relevance"]["score"]}/5', ev["answer_relevance"]["severity"])
    with st.expander("Why (judge reasoning + findings)"):
        st.write("**Faithfulness:**", ev["faithfulness"]["reason"])
        st.write("**Answer relevance:**", ev["answer_relevance"]["reason"])
        if ev["findings"]:
            st.write("**Findings:**")
            for f in ev["findings"]:
                st.write("-", f)
