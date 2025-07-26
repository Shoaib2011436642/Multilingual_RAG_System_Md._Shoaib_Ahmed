# Multilingual RAG System (Bangla & English) with Evaluation

## 1. Setup Guide

### Windows (PowerShell)

1. **Open PowerShell** in the directory where you have downloaded the RAG system.

2. **Check PowerShell execution policy:**

   ```powershell
   Get-ExecutionPolicy
   ```

   - If it shows **Restricted**, temporarily bypass it:
     ```powershell
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     ```

3. **Create and activate a virtual environment:**

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, install manually:

   ```powershell
   pip install flask langchain langchain_huggingface langchain_community langchain_openai sentence-transformers faiss-cpu numpy scikit-learn pymupdf
   ```

   (Additional packages may include `requests` if needed.)

5. **Change FAISS vector store path in **``**:**

   Locate this line (around line 17):

   ```python
   vectorstore = FAISS.load_local(
       "E:/RAG_Final_Shoaib/bangla_vector_db",
       embeddings=embedding_model,
       allow_dangerous_deserialization=True
   )
   ```

   Replace it with your own directory:

   ```python
   vectorstore = FAISS.load_local(
       "<your_downloaded_directory>/bangla_vector_db",
       embeddings=embedding_model,
       allow_dangerous_deserialization=True
   )
   ```

6. **Run the Flask app:**

   ```powershell
   python app.py
   ```

7. After you see:

   ```
   * Debugger is active!
   * Debugger PIN: 910-365-151
   ```

   open the following URL in your browser:

   ```
   http://127.0.0.1:5000/
   ```

---
## Warning:

As I have used the LLM from Open Router, which provides free api keys. It might show limit error as the usage limit is present in free models in open router. :")

## 2. Folder Structure

```
RAG_Final_Shoaib/
│
├── app.py                  # Flask application (with evaluation)
├── index.html              # Frontend template (in /templates folder)
├── /templates/
│   └── index.html
├── /bangla_vector_db/      # FAISS vector store files
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

**Note:** `venv/` is excluded because GitHub repositories typically do not store virtual environments. Create your own venv as shown above.

---

## 3. Tools, Libraries, and Packages Used

- **Python packages:**

  - Flask
  - langchain
  - langchain\_huggingface
  - langchain\_community
  - langchain\_openai
  - sentence-transformers
  - FAISS (faiss-cpu)
  - NumPy
  - scikit-learn
  - PyMuPDF (fitz)

- **LLM API:**

  - OpenRouter (model: `mistralai/mistral-7b-instruct:free`)

- **Development Tools:**

  - **Google Colab** → Preprocessing, vector database creation, embeddings.
  - **PyCharm** → Flask app development & evaluation code.

- **Reference:**

  - [Colab Notebook](https://colab.research.google.com/drive/1Ye5VaW3nYDgxyvGcSD6tbXbs3aIgvIQh?usp=sharing)

---

## 4. Sample Queries & Outputs

### Bangla Queries:

1. `কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?`
2. `বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?`
3. `অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?`

### English Queries:

1. `Who was referred to as Anupam's destiny god?`
2. `What was Kalyani's actual age at the time of marriage?`

---

## 5. API Documentation

- **API Key:** OpenRouter (`OPENAI_API_KEY`)
- **Model Used:** `mistralai/mistral-7b-instruct:free`
- **Base URL:** `https://openrouter.ai/api/v1`
- **Endpoint:**
  - `/` → Loads the main page.
  - `/ask` (POST) → Accepts user question → returns answer + evaluation scores.

---

## 6. Evaluation Matrix

**Implemented:**

- **Relevance:**
  - Average & maximum cosine similarity between query embeddings and retrieved documents.
- **Groundedness:**
  - Word-overlap ratio between generated answer and retrieved documents.

Evaluation scores are **displayed dynamically in the web UI** after submitting a query.

---

## 7. Key Questions

### ✅ What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

We used **PyMuPDF (fitz)** because it can extract Unicode text while preserving most formatting. However, Bangla PDFs often contain encoding inconsistencies, broken ligatures, and spacing issues. Converting Bangla PDFs to `.txt` first reduced these problems significantly.

### ✅ What chunking strategy did you choose? Why does it work?

We used **character-based chunking with overlaps**. This keeps each chunk small enough for the embedding model while maintaining context continuity. Overlaps prevent information loss and make semantic retrieval more accurate.

### ✅ What embedding model did you use? Why?

We selected `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` because:

- It supports **both Bangla and English**.
- It's lightweight, fast, and well-suited for FAISS storage.
- It captures semantic meaning beyond simple keyword matches, improving retrieval quality.

### ✅ How are you comparing the query with stored chunks? Why this method?

We use **cosine similarity** between query embeddings and chunk embeddings. Cosine similarity is the standard for semantic search because it measures **direction (semantic closeness)** rather than vector magnitude. **FAISS** accelerates similarity search for large datasets.

### ✅ How do you ensure meaningful comparison between query and chunks? What if the query is vague?

Both queries and chunks are embedded in the same semantic vector space. If a query is vague, FAISS will still return the closest matches, but accuracy may suffer. To improve vague-query handling, one could add **query expansion, prompt optimization, or larger context retrieval**.

### ✅ Are results relevant? How to improve them?

Some results are relevant, but accuracy drops when:

- PDF text extraction is noisy.
- Chunks are too small or lack context.
- Bangla language nuances affect embeddings.

Improvements:

- Use **better PDF preprocessing** or clean text datasets.
- Increase chunk size or use **sentence-based chunking with overlap**.
- Try **larger multilingual embedding models (e.g., LaBSE, mUSE)**.

---

