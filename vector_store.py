# vector_store.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def create_vector_store(chunks, persist_path: str = "faiss_index"):
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore

def load_vector_store(persist_path: str = "faiss_index"):
    embeddings = get_embedding_model()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)