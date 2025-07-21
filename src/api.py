from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from rank_bm25 import BM25Okapi

app = FastAPI()
model_name = "deepset/roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# RAG components
EMBED_MODEL = 'sentence-transformers/all-mpnet-base-v2'  # upgraded model
INDEX_PATH = 'models/rag_faiss.index'
CHUNKS_PATH = 'models/rag_chunks.pkl'
embedder = SentenceTransformer(EMBED_MODEL)
faiss_index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, 'rb') as f:
    doc_chunks = pickle.load(f)

# BM25 index
bm25_corpus = [chunk.split() for chunk in doc_chunks]
bm25 = BM25Okapi(bm25_corpus)

class QARequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Software Documentation QA System (RAG Hybrid)"}

@app.post("/query")
def query_qa(request: QARequest):
    try:
        # Embedding retrieval
        q_emb = embedder.encode([request.question], convert_to_numpy=True)
        D, I = faiss_index.search(q_emb, k=20)  # top-20
        emb_chunks = [doc_chunks[idx] for idx in I[0]]
        # BM25 retrieval
        bm25_scores = bm25.get_scores(request.question.split())
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
        bm25_chunks = [doc_chunks[i] for i in bm25_top_indices]
        # Merge and deduplicate
        candidate_contexts = list(dict.fromkeys(emb_chunks + bm25_chunks))
        print("\n--- Top 20 Embedding Chunks ---")
        for i, ctx in enumerate(emb_chunks, 1):
            print(f"Emb Chunk {i}:\n{ctx}\n{'-'*40}")
        print("\n--- Top 20 BM25 Chunks ---")
        for i, ctx in enumerate(bm25_chunks, 1):
            print(f"BM25 Chunk {i}:\n{ctx}\n{'-'*40}")
        best_answer = ''
        best_score = float('-inf')
        best_context = ''
        for context in candidate_contexts:
            if len(context.strip()) < 50:  # skip very short chunks
                continue
            max_length = 512  # for roberta-large
            inputs = tokenizer(
                request.question,
                context,
                return_tensors="pt",
                max_length=max_length,
                truncation="only_second"
            )
            with torch.no_grad():
                outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            answer_start = torch.argmax(start_logits)
            answer_end = torch.argmax(end_logits) + 1
            score = start_logits[0, answer_start] + end_logits[0, answer_end-1]
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
            )
            if score > best_score and answer.strip() and answer.strip() != '<s>':
                best_score = score
                best_answer = answer.strip()
                best_context = context
        if not best_answer:
            return {"answer": "No answer found in documentation.", "context": ""}
        return {"answer": best_answer, "context": best_context}
    except Exception as e:
        return {"error": str(e)}
