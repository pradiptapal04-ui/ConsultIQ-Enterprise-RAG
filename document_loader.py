import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_document(file_path: str):
    """Loads text from PDF or TXT files."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return loader.load()