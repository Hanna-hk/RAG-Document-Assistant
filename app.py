import streamlit as st
from ingest import process_pdf
from rag import generate_answer

st.title("RAG-Assistant for document analysis")

uploaded_file = st.file_uploader("Load PDF", type="pdf")
if uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    process_pdf(uploaded_file.name)
    st.success("Document has been loaded to the base!")

query = st.text_input("Ask a question based on the documents:")
if query:
    with st.spinner("Searching for the answer..."):
        answer, sources = generate_answer(query)
    st.write(answer)
    with st.expander("Sources"):
        for s in sources:
            st.write(f"- {s['metadata']['source']} (similarity: {s['similarity']:.2f})")