from flask import Flask, request, jsonify, render_template
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI  # or your LLM class
from langchain_mistralai import ChatMistralAI
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = FAISS.load_local(
    "E:/RAG_Final_Shoaib/bangla_vector_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Create retriever and QA chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

from langchain_openai import ChatOpenAI
import os

# Set Mistral API key
os.environ["MISTRAL_API_KEY"] = "FbLAJBFsSnqIjydjm3W2DHfsdzpVJq14"

llm = ChatMistralAI(
    mistral_api_key=os.environ["MISTRAL_API_KEY"],
    model="mistral-medium"  # or "mistral-medium", depending on availability
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)  # NOTE: return docs now!

def evaluate_relevance(query, retrieved_docs, embedding_model):
    query_embedding = embedding_model.embed_query(query)
    doc_embeddings = [embedding_model.embed_query(doc.page_content) for doc in retrieved_docs]
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    avg_sim = np.mean(similarities)
    max_sim = np.max(similarities)
    return avg_sim, max_sim

def evaluate_groundedness(answer, retrieved_docs):
    combined_docs_text = " ".join([doc.page_content for doc in retrieved_docs])
    answer_words = set(answer.split())
    doc_words = set(combined_docs_text.split())
    overlap = answer_words.intersection(doc_words)
    groundedness_score = len(overlap) / len(answer_words) if answer_words else 0
    return groundedness_score

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    query = request.form.get("query")
    try:
        result = qa_chain({"query": query})
        answer = result['result']
        retrieved_docs = result['source_documents']

        # Compute evaluation scores
        relevance_avg, relevance_max = evaluate_relevance(query, retrieved_docs, embedding_model)
        groundedness = evaluate_groundedness(answer, retrieved_docs)

    except Exception as e:
        if "Rate limit exceeded" in str(e):
            answer = "Sorry, you hit the rate limit for OpenAI. Please try later. (As it is free version for LLM API Keys and it has limits.)"
            relevance_avg = relevance_max = groundedness = None
        else:
            raise e

    return render_template(
        "index.html",
        query=query,
        answer=answer,
        relevance_avg=round(relevance_avg, 3) if relevance_avg is not None else None,
        relevance_max=round(relevance_max, 3) if relevance_max is not None else None,
        groundedness=round(groundedness, 3) if groundedness is not None else None,
    )

if __name__ == "__main__":
    app.run(debug=True)