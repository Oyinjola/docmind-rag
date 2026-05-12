"""
RAG (Retrieval-Augmented Generation) Engine
Ingests documents → chunks → embeds → stores in vector index → retrieves on query
No external vector DB needed — pure NumPy FAISS-style cosine similarity
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional
import hashlib


# ── Text Chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    """Split text into overlapping chunks, preserving sentence boundaries."""
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks    = []
    current   = []
    current_len = 0
    
    for sent in sentences:
        sent_len = len(sent.split())
        
        if current_len + sent_len > chunk_size and current:
            chunk_text_str = ' '.join(current)
            chunks.append({
                'text':       chunk_text_str,
                'word_count': current_len,
                'chunk_id':   len(chunks),
            })
            # Keep overlap
            overlap_words = ' '.join(current).split()[-overlap:]
            current       = [' '.join(overlap_words)]
            current_len   = len(overlap_words)
        
        current.append(sent)
        current_len += sent_len
    
    if current:
        chunks.append({
            'text':       ' '.join(current),
            'word_count': current_len,
            'chunk_id':   len(chunks),
        })
    
    return chunks


# ── TF-IDF Vectorizer (no sklearn) ────────────────────────────────────────────
class TFIDFVectorizer:
    """Bag-of-words TF-IDF from scratch."""

    def __init__(self, max_features: int = 2000, min_df: int = 1):
        self.max_features = max_features
        self.min_df       = min_df
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray       = np.array([])

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Remove stopwords
        stops = {'the','a','an','is','it','in','on','at','to','of','and','or',
                 'for','be','was','are','were','has','have','had','this','that',
                 'with','as','by','from','its','not','but','if','so','do','does'}
        return [t for t in tokens if t not in stops and len(t) > 2]

    def fit(self, docs: list[str]) -> 'TFIDFVectorizer':
        n = len(docs)
        df_count: dict[str, int] = {}

        for doc in docs:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df_count[t] = df_count.get(t, 0) + 1

        # Filter by min_df, sort by df desc, take top max_features
        vocab = {t: c for t, c in df_count.items() if c >= self.min_df}
        vocab = sorted(vocab.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocab_ = {t: i for i, (t, _) in enumerate(vocab)}

        # IDF
        self.idf_ = np.zeros(len(self.vocab_))
        for t, i in self.vocab_.items():
            self.idf_[i] = np.log((n + 1) / (df_count.get(t, 0) + 1)) + 1

        return self

    def transform(self, docs: list[str]) -> np.ndarray:
        V = len(self.vocab_)
        mat = np.zeros((len(docs), V), dtype=np.float32)

        for d_idx, doc in enumerate(docs):
            tokens  = self._tokenize(doc)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            n_tokens = len(tokens) or 1
            for t, cnt in tf.items():
                if t in self.vocab_:
                    i = self.vocab_[t]
                    mat[d_idx, i] = (cnt / n_tokens) * self.idf_[i]

        # L2 normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        return mat / norms

    def fit_transform(self, docs: list[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)


# ── Vector Store ──────────────────────────────────────────────────────────────
class VectorStore:
    """In-memory cosine similarity search over TF-IDF embeddings."""

    def __init__(self):
        self.chunks:    list[dict]  = []
        self.embeddings: np.ndarray = np.array([])
        self.vectorizer = TFIDFVectorizer(max_features=3000)
        self._fitted    = False

    def add_document(self, text: str, metadata: Optional[dict] = None) -> int:
        """Chunk and index a document. Returns number of chunks added."""
        new_chunks = chunk_text(text)
        doc_id     = hashlib.md5(text[:200].encode()).hexdigest()[:8]
        
        for c in new_chunks:
            c['doc_id']   = doc_id
            c['metadata'] = metadata or {}
        
        self.chunks.extend(new_chunks)
        self._refit()
        return len(new_chunks)

    def _refit(self):
        texts            = [c['text'] for c in self.chunks]
        self.embeddings  = self.vectorizer.fit_transform(texts)
        self._fitted     = True

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k most relevant chunks for query."""
        if not self._fitted or len(self.chunks) == 0:
            return []
        
        q_vec = self.vectorizer.transform([query])           # (1, V)
        sims  = (self.embeddings @ q_vec.T).flatten()        # cosine sim
        
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in top_idx:
            if sims[i] > 0.01:   # relevance threshold
                results.append({
                    **self.chunks[i],
                    'score': float(sims[i])
                })
        return results

    def clear(self):
        self.chunks     = []
        self.embeddings = np.array([])
        self._fitted    = False

    def stats(self) -> dict:
        doc_ids = {c['doc_id'] for c in self.chunks}
        return {
            'total_chunks': len(self.chunks),
            'total_docs':   len(doc_ids),
            'vocab_size':   len(self.vectorizer.vocab_) if self._fitted else 0,
        }


# ── RAG Pipeline ──────────────────────────────────────────────────────────────
class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    Uses VectorStore for retrieval, OpenAI for generation.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        self.store   = VectorStore()
        self.api_key = openai_api_key
        self.history: list[dict] = []

    def ingest(self, text: str, source_name: str = "document") -> dict:
        n_chunks = self.store.add_document(text, {'source': source_name})
        stats    = self.store.stats()
        return {
            'source':   source_name,
            'chunks':   n_chunks,
            'total':    stats['total_chunks'],
            'vocab':    stats['vocab_size'],
        }

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        return self.store.search(query, top_k=top_k)

    def build_prompt(self, query: str, chunks: list[dict]) -> tuple[str, str]:
        context_parts = []
        for i, c in enumerate(chunks):
            src = c['metadata'].get('source', 'doc')
            context_parts.append(f"[Source {i+1}: {src}]\n{c['text']}")
        
        context = "\n\n---\n\n".join(context_parts)

        system = (
            "You are a precise, helpful document assistant. "
            "Answer the user's question using ONLY the provided context. "
            "If the answer is not in the context, say so clearly. "
            "Cite [Source N] when referencing specific information. "
            "Be concise and factual."
        )

        user = f"Context:\n{context}\n\nQuestion: {query}"
        return system, user

    def answer(self, query: str, top_k: int = 4) -> dict:
        """Full RAG cycle: retrieve → augment prompt → generate answer."""
        chunks = self.retrieve(query, top_k=top_k)

        if not chunks:
            return {
                'answer':   "No relevant content found in the uploaded documents.",
                'sources':  [],
                'chunks':   [],
                'fallback': True,
            }

        system, user = self.build_prompt(query, chunks)

        # Add to conversation history
        self.history.append({"role": "user", "content": user})

        # Call OpenAI API
        import urllib.request
        payload = json.dumps({
            "model":    "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                *self.history[-6:]   # last 3 turns
            ],
            "max_tokens":  800,
            "temperature": 0.2,
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key or ''}",
            },
            method = "POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            answer = result['choices'][0]['message']['content']
        except Exception as e:
            # Graceful fallback — return retrieved context summary
            answer = (
                f"[API unavailable — showing retrieved context]\n\n"
                + "\n\n".join(c['text'][:300] + "..." for c in chunks[:2])
            )

        self.history.append({"role": "assistant", "content": answer})

        return {
            'answer':  answer,
            'sources': list({c['metadata'].get('source','doc') for c in chunks}),
            'chunks':  [{'text': c['text'][:200] + '...', 'score': round(c['score'], 4),
                         'source': c['metadata'].get('source','doc')} for c in chunks],
        }

    def reset_history(self):
        self.history = []


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGPipeline()

    sample = """
    Machine learning is a subfield of artificial intelligence that enables systems to learn
    from data without being explicitly programmed. Supervised learning uses labeled training
    data to build predictive models. Common algorithms include linear regression for continuous
    outputs, logistic regression for classification, and decision trees which partition the
    feature space recursively.

    Deep learning uses neural networks with multiple hidden layers to learn complex patterns.
    Convolutional neural networks (CNNs) excel at image tasks. Transformers, introduced in
    the paper Attention Is All You Need, revolutionized NLP by processing sequences in parallel
    using self-attention mechanisms. Large language models like GPT are trained on vast text
    corpora using next-token prediction as the training objective.

    Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative model.
    Documents are chunked, embedded into a vector space, and stored in a vector database.
    At inference time, the query is embedded and the most similar chunks are retrieved,
    then injected into the prompt as context for the language model to ground its response.
    """

    result = rag.ingest(sample, "ml_notes.txt")
    print(f"Ingested: {result}")

    queries = [
        "What is retrieval-augmented generation?",
        "How do transformers work?",
        "What is supervised learning?",
    ]

    for q in queries:
        r = rag.answer(q)
        print(f"\nQ: {q}")
        print(f"Sources: {r['sources']}")
        print(f"A: {r['answer'][:300]}...")