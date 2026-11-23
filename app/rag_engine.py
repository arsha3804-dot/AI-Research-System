import os
import glob
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from app.config import settings


class LocalRAGEngine:
    """
    100% Free Local RAG Engine using ChromaDB + HuggingFace SentenceTransformers.
    Zero API cost for embeddings or vector storage.
    """

    def __init__(self):
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        os.makedirs(settings.docs_dir, exist_ok=True)

        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="research_knowledge_base",
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _chunk_text(self, text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += max(1, chunk_size - overlap)
        return chunks

    def ingest_text(self, doc_id_prefix: str, title: str, source: str, text: str) -> int:
        chunks = self._chunk_text(text)
        if not chunks:
            return 0
        ids = [f"{doc_id_prefix}_chunk_{idx}" for idx in range(len(chunks))]
        metadatas = [
            {"title": title, "source": source, "chunk_index": idx}
            for idx in range(len(chunks))
        ]
        self.collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    def ingest_directory(self) -> Dict[str, Any]:
        total_chunks = 0
        files_processed = []
        patterns = ["*.txt", "*.md", "*.pdf"]
        for pattern in patterns:
            for filepath in glob.glob(os.path.join(settings.docs_dir, pattern)):
                filename = os.path.basename(filepath)
                text = ""
                if filepath.endswith(".pdf"):
                    reader = PdfReader(filepath)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                else:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()

                count = self.ingest_text(
                    doc_id_prefix=filename,
                    title=filename,
                    source=filepath,
                    text=text,
                )
                total_chunks += count
                files_processed.append({"file": filename, "chunks": count})

        return {"files_processed": files_processed, "total_chunks": total_chunks}

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        n_results = min(top_k, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=n_results)

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(
                {
                    "title": meta.get("title", "Internal Doc"),
                    "source": meta.get("source", "local_kb"),
                    "snippet": doc,
                    "similarity": round(1.0 - float(dist), 4),
                }
            )
        return hits


rag_engine = LocalRAGEngine()
