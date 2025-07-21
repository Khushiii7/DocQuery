# **DocQuery** - Hugging Face Transformers Documentation QA System
A Retrieval-Augmented Generation (RAG) system to answer developer questions about the Hugging Face Transformers library documentation. Built with FastAPI, Streamlit and state-of-the-art NLP models.
This project enables users to ask natural language questions about the Hugging Face Transformers documentation and get accurate, context-supported answers. It combines document scraping, preprocessing, semantic chunking, vector search (FAISS) and a powerful QA model.


## 🛠️ Tech Stack
- **Python 3.8+**
- **FastAPI** (API backend)
- **Streamlit** (frontend)
- **Hugging Face Transformers**
- **Sentence Transformers**
- **FAISS** (vector search)
- **rank_bm25** (keyword search)
- **BeautifulSoup, requests** (scraping)
- **Docker** (optional, for deployment)

## ⚡ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/transformers-doc-qa.git
cd transformers-doc-qa
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Scrape and Preprocess Documentation
```bash
python src/scraping.py
python src/preprocessing.py
```

### 4. Build the RAG Index
```bash
python src/rag_build_index.py
```

### 5. Start the FastAPI Backend
```bash
uvicorn src.api:app --reload
```

### 6. Launch the Streamlit Frontend
```bash
streamlit run streamlit_app.py
```
