from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

def detect_title_match(question, documents):
    question_lower = question.lower()

    for doc in documents:
        title = doc.metadata.get("title", "").lower()

        if title and title in question_lower:
            return doc
        
    return None


def load_doc_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    doc_store = FAISS.load_local(
        "../doc_vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return doc_store.as_retriever(search_kwargs={"k": 5})


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

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model once
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def classify_question(question):
    question_lower = question.strip().lower()

    if question_lower.startswith(("when", "who", "where", "created")):
        return "factoid"
    elif any(word in question_lower for word in ["list", "procedures", "steps", "types", "categories"]):
        return "list"
    else:
        return "general"

def extract_sentences_from_chunks(chunks):
    sentences = []
    for chunk in chunks:
        split_sentences = re.split(r'(?<=[.!?])\s+', chunk.page_content)
        sentences.extend([s.strip() for s in split_sentences if len(s.strip()) > 20])
    return sentences

def extractive_answer(question, chunks):

    # Step 1: Classify question type
    q_type = classify_question(question)

    # Step 2: Decide top_n
    if q_type == "factoid":
        top_n = 1
    elif q_type == "list":
        top_n = 5
    else:
        top_n = 3

    # Step 3: Extract sentences
    sentences = extract_sentences_from_chunks(chunks)

    if not sentences:
        return "The information is not mentioned in the provided documents."

    # Step 4: Embed question and sentences
    question_embedding = embedding_model.embed_query(question)
    sentence_embeddings = embedding_model.embed_documents(sentences)

    # Step 5: Compute similarity
    similarities = cosine_similarity(
        [question_embedding],
        sentence_embeddings
    )[0]

    # Step 6: Rank sentences
    ranked_indices = np.argsort(similarities)[::-1]

    # Step 7: Apply threshold
    threshold = 0.6# You can tune this
    selected_sentences = []

    #for i in ranked_indices:
        #print(f"Sentence: {sentences[i][:60]}... | Similarity: {similarities[i]:.4f}")
    for idx in ranked_indices:
        if similarities[idx] < threshold:
            break
        selected_sentences.append(sentences[idx])
        if len(selected_sentences) >= top_n:
            break

    if not selected_sentences:
        return "The information is not mentioned in the provided documents."

    return "\n\n".join(selected_sentences)#, similarities,sentences



def answer_question(question):

    # Stage 1 — Detect document
    #doc = doc_retriever.invoke(question)[0]
    # Load all documents from doc_store
    all_docs = doc_retriever.vectorstore.docstore._dict.values()

    # Try direct title match first
    matched_doc = detect_title_match(question, all_docs)

    if matched_doc:
        doc = matched_doc
        print("Title Match Found:", doc.metadata["doc_id"])
    else:
        doc = doc_retriever.invoke(question)[0]
        print("Semantic Match Used:", doc.metadata["doc_id"])
    
    doc_id = doc.metadata["doc_id"]

    print("Detected Document:", doc_id)

    # Stage 2 — Retrieve chunks from detected doc
    chunks = chunk_store.similarity_search(
        question,
        k=6,
        filter={"doc_id": doc_id}
    )

    # Stage 3 — Extractive ranking
    final_answer = extractive_answer(question, chunks)
    return final_answer, doc_id

'''def answer_question(question):

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

    context = "\n\n".join(chunk.page_content for chunk in chunks)'''
 

if __name__ == "__main__":
    question = "From which department does this Investigating the Efficacy of Pembrolizumab in Combination with Chemotherapy for the Treatment of Advanced Non-Small Cell Lung Cancer (NSCLC) belong ?"
    answer = answer_question(question)
    print(answer)

