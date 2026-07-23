"""
RAG-Assistant for document analysis.

A Streamlit web application that lets a user upload a PDF document,
ingest it into a vector database (Supabase/pgvector) via `ingest.process_pdf`,
and then ask natural-language questions about its content. Answers are
generated using retrieval-augmented generation (RAG) via `rag.generate_answer`,
with the source documents and similarity scores displayed alongside each answer.

Uploaded files are written to a secure temporary file to avoid path traversal or
overwriting existing files, and the temp file is removed after processing
regardless of success or failure.

Run with:
    streamlit run app.py
"""
import streamlit as st
from ingest import process_pdf
from rag import generate_answer
import tempfile
import os

st.title("RAG-Assistant for document analysis")

uploaded_file = st.file_uploader("Load PDF", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        process_pdf(tmp_path, original_name=uploaded_file.name)
        st.success("Document has been loaded to the base!")
    except Exception as e:
        st.error(f"Failed to process document: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

query = st.text_input("Ask a question based on the documents:")
if query:
    with st.spinner("Searching for the answer..."):
        answer, sources = generate_answer(query)
    st.write(answer)
    with st.expander("Sources"):
        for s in sources:
            st.write(f"- {s['metadata']['source']} (similarity: {s['similarity']:.2f})")