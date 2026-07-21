import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv
from utils import chunk_text

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

def process_pdf(filepath):
    filename = os.path.basename(filepath)

    supabase.table("documents").delete().eq("metadata->>source", filename).execute()
    reader = PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = chunk_text(full_text.replace("\x00", ""))
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