import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks=[]
    for i in range (0, len(words), chunk_size-overlap):
        chunk = " ". join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdf(filepath):
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = chunk_text(full_text)
    embeddings = model.encode(chunks).tolist()
    rows = [
        {
            "content":chunk,
            "metadata":{"source": os.path.basename(filepath)},
            "embedding": emb
        }
        for chunk, emb in zip(chunks, embeddings)
    ]

    supabase.table("documents").insert(rows).execute()
    print(f"Downloaded {len(rows)} chunks from {filepath}")

if __name__ == "__main__":
    process_pdf("example.pdf")