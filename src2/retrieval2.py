from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_doc_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    doc_store = FAISS.load_local(
        "../doc_vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return doc_store.as_retriever(search_kwargs={"k": 1})


def load_chunk_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "../chunk_vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

doc_retriever = load_doc_retriever()
chunk_store = load_chunk_store()

def answer_question(question):

    # 🔹 Stage 1 — Detect document
    doc = doc_retriever.invoke(question)[0]
    doc_id = doc.metadata["doc_id"]

    print("Detected Document:", doc_id)

    # 🔹 Stage 2 — Search only inside that document
    chunks = chunk_store.similarity_search(
        question,
        k=6,
        filter={"doc_id": doc_id}
    )

    context = "\n\n".join(chunk.page_content for chunk in chunks)

    return context

if __name__ == "__main__":
    question = "What are the data handling procedures for Clinical Protocol for Patient Confidentiality and Data Protection?"
    answer = answer_question(question)
    print(answer)