from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ingestion2 import load_documents

def create_vector_stores(documents):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔹 1️⃣ Document-Level Vector Store
    doc_store = FAISS.from_documents(documents, embeddings)
    doc_store.save_local("../doc_vector_db")

    # 🔹 2️⃣ Chunk-Level Vector Store
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunk_docs = splitter.split_documents(documents)

    chunk_store = FAISS.from_documents(chunk_docs, embeddings)
    chunk_store.save_local("../chunk_vector_db")

if __name__ == "__main__":
    docs = load_documents()
    create_vector_stores(docs)