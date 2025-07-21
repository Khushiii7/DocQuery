import os
import glob
import pickle
import re
from sentence_transformers import SentenceTransformer
import faiss

# Parameters
PROCESSED_DIR = 'data/processed/'
INDEX_PATH = 'models/rag_faiss.index'
CHUNKS_PATH = 'models/rag_chunks.pkl'
EMBED_MODEL = 'sentence-transformers/all-mpnet-base-v2'  # upgraded model

# Helper: Clean navigation/boilerplate lines
NAVIGATION_KEYWORDS = [
    '🏡 View all docs', 'AWS Trainium', 'Chat UI', 'Hub', 'Leaderboards', 'Sign Up',
    'Search documentation', 'main', 'v4.', 'Tasks', 'Transformers documentation',
    'Transformers.js', 'timm', 'smolagents', 'Optimum', 'PEFT', 'Sentence Transformers',
    'Tokenizers', 'Datasets', 'Diffusers', 'Evaluate', 'Gradio', 'Hub Python Library',
    'Huggingface.js', 'Inference Endpoints', 'Inference Providers', 'LeRobot', 'Lighteval',
    'Microsoft Azure', 'Safetensors', 'TRL', 'Text Embeddings Inference', 'Text Generation Inference',
    'AutoTrain', 'Bitsandbytes', 'Dataset viewer', 'Distilabel', 'Argilla', 'Deploying on AWS',
    'Processed', 'Raw', 'Installation', 'Export to production', 'Resources', 'Contribute', 'API',
    'Base classes', 'Inference', 'Training', 'Quantization', 'Export to production', 'Models',
    'Internal helpers', 'Reference', 'Join the Hugging Face community', 'Collaborate on models',
    'Faster examples with accelerated inference', 'Switch between documentation themes',
    'Sign Up', 'to get started', 'Copied', 'source', 'Update on GitHub', 'EN', 'DE', 'FR', 'ES',
    'IT', 'JA', 'KO', 'PT', 'TE', 'TR', 'ZH', 'AR', 'HI', 'v4.', 'doc-builder-html'
]

def is_navigation_line(line):
    return any(kw in line for kw in NAVIGATION_KEYWORDS) or not line.strip()

def chunk_paragraphs(paragraphs, min_length=20):
    return [p.strip() for p in paragraphs if len(p.strip()) > min_length]

chunks = []
file_chunk_counts = {}
for file in glob.glob(os.path.join(PROCESSED_DIR, '*.txt')):
    with open(file, encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line for line in lines if not is_navigation_line(line)]
        text = ''.join(lines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        file_chunks = chunk_paragraphs(paragraphs, min_length=20)
        chunks.extend(file_chunks)
        file_chunk_counts[file] = len(file_chunks)

print(f"Number of files processed: {len(file_chunk_counts)}")
for fname, count in file_chunk_counts.items():
    print(f"{fname}: {count} chunks")
print(f"Total number of chunks: {len(chunks)}")

if len(chunks) == 0:
    raise ValueError("No valid chunks were created. Check your processed files and chunking logic.")

# 2. Embed chunks
model = SentenceTransformer(EMBED_MODEL)
embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
print(f"Embeddings shape: {embeddings.shape}")

if embeddings.shape[0] == 0 or len(embeddings.shape) != 2:
    raise ValueError("Embeddings are empty or not 2D. Check chunking and embedding model.")

# 3. Build FAISS index
os.makedirs('models', exist_ok=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 4. Save index and chunks
with open(CHUNKS_PATH, 'wb') as f:
    pickle.dump(chunks, f)

faiss.write_index(index, INDEX_PATH)
print(f"Indexed {len(chunks)} chunks. Index saved to {INDEX_PATH}, chunks to {CHUNKS_PATH}.") 