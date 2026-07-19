import os
from supabase import create_client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def search_context(query, match_count=5, threshold=0.3):
    query_embedding = model.encode([query])[0].tolist()
    response = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_threshold": threshold,
        "match_count": match_count
    }).execute()
    return response.data

def generate_answer(query):
    matches = search_context(query)
    if not matches:
        return "There is no information in documents", []

    context = "\n\n---\n\n".join(
        [f"[Resource: {m['metadata']['source']}]\n{m['content']}" for m in matches])
    system_prompt = ("""
        You are an assistant, that answers STRICTLY based on given context.
        If there is no answer in context - truly say about that.
        Always give a source (file), from which the information was taken.
        """)

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
        ]
    )
    return completion.choices[0].message.content, matches