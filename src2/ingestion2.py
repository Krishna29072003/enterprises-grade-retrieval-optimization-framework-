from langchain_core.documents import Document
import os
#DATA_PATH = "../data/raw_docs"
DATA_PATH = "../src/data/raw_docs"

def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            first_line = text.split("\n")[0]
            title = first_line.replace("Title: ", "").strip()

        doc_id = file.replace(".txt", "")

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": doc_id,
                    "title": title
                }
            )
        )

    return documents

print("Looking inside:", os.path.abspath(DATA_PATH))
print('Exists?', os.path.exists(DATA_PATH))