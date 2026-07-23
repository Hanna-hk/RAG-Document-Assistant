import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv
from utils import chunk_text

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

def process_pdf(filepath, original_name=None):
    """
        Parse a PDF file, split its text into overlapping chunks, generate
        embeddings for each chunk, and store the results in Supabase.

        Any existing chunks previously stored for the same source filename
        are deleted first, so re-ingesting a document replaces its old data
        instead of duplicating it.

        Args:
            filepath (str): Path to the PDF file on disk to read from.
            original_name (str, optional): Human-readable filename to store
                in the metadata (e.g. the name the user originally uploaded).
                Falls back to the basename of `filepath` if not provided.
                Useful when `filepath` points to a temporary file with a
                randomly generated name.

        Returns:
            None
        """
    filename = original_name or os.path.basename(filepath)

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