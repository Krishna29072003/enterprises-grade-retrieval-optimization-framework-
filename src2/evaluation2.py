from retrieval2 import answer_question  # Make sure this returns (answer, doc_id)

# ----------------------------
# 1️⃣ Define Evaluation Set
# ----------------------------

evaluation_set = [
    {
        "question": "When was Clinical Protocol for Patient Confidentiality and Data Protection created?",
        "expected_doc_id": "HC_045",
        "expected_answer_contains": ["01 February 2024"]
    },
    {
        "question": "What is the background and context for Analysis of Telemedicine Adoption in Rural Healthcare Settings?",
        "expected_doc_id": "HC_024",
        "expected_answer_contains": ["Rural healthcare settings face unique challenges, including limited access to specialized healthcare services, shortage of healthcare professionals, and limited healthcare infrastructure. Telemedicine has been proposed as a potential solution to address these challenges by providing remote access to specialized healthcare services. However, the adoption of telemedicine in rural healthcare settings has been slow due to various barriers, including limited infrastructure, lack of digital literacy, and regulatory challenges. This study aims to analyze the adoption of telemedicine in rural healthcare settings, identify the barriers to adoption, and propose strategies to overcome these barriers."]
    },
    {
        "question": "What is the department of Clinical Protocol for Patient Confidentiality and Data Protection?",
        "expected_doc_id": "HC_045",
        "expected_answer_contains": ["Department:Administration"]
    },
    {
        "question": "What is the purpose and scopeof Management of Metastatic Colorectal Cancer with Targeted Therapies?",
        "expected_doc_id": "HC_050",
        "expected_answer_contains": ["This clinical protocol outlines the management of metastatic colorectal cancer (mCRC) with targeted therapies in the Oncology Department of our healthcare institution. The purpose of this protocol is to provide a standardized approach to the diagnosis, treatment, and follow-up of patients with mCRC, ensuring optimal patient outcomes and minimizing treatment-related toxicities."]
    },
    {
        "question": "what is the purpose of Compliance Manual for Emergency Department?",
        "expected_doc_id": "HC_038",
        "expected_answer_contains": ["This Compliance Manual for the Emergency Department outlines the policies, procedures, and guidelines that must be followed to ensure compliance with relevant laws, regulations, and industry standards. The purpose of this manual is to provide a framework for emergency department staff to understand their roles and responsibilities in maintaining compliance with applicable laws and regulations."]
    }
]


# ----------------------------
# 2️⃣ Evaluation Logic
# ----------------------------

def evaluate_system(evaluation_set):

    total_questions = len(evaluation_set)

    doc_correct = 0
    answer_correct = 0

    for item in evaluation_set:
        question = item["question"]
        expected_doc = item["expected_doc_id"]
        expected_keywords = item["expected_answer_contains"]

        print("\n-----------------------------------")
        print("Question:", question)

        # IMPORTANT: Your answer_question must return BOTH answer and detected_doc
        answer, detected_doc = answer_question(question)

        print("Detected Doc:", detected_doc)
        print("Answer:\n", answer)

        # 🔹 Document Accuracy
        if detected_doc == expected_doc:
            doc_correct += 1

        # 🔹 Answer Hit Check
        if any(keyword.lower() in answer.lower() for keyword in expected_keywords):
            answer_correct += 1

    # ----------------------------
    # 3️⃣ Final Metrics
    # ----------------------------

    document_accuracy = doc_correct / total_questions
    answer_hit_rate = answer_correct / total_questions

    print("\n===================================")
    print("Document Accuracy:", round(document_accuracy, 3))
    print("Answer Hit Rate:", round(answer_hit_rate, 3))
    print("===================================")


# ----------------------------
# 4️⃣ Run Evaluation
# ----------------------------

if __name__ == "__main__":
    evaluate_system(evaluation_set)