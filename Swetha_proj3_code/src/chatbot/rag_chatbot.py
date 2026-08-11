from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.loader import load_rag_corpus


class RAGChatbot:
    """TF-IDF RAG chatbot over project knowledge corpus."""

    def __init__(self, corpus_path: str | Path):
        self.chunks = load_rag_corpus(corpus_path)
        texts = [f"{c['title']}. {c['content']}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    def query(self, question: str, top_k: int = 3) -> dict:
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            c = self.chunks[i]
            results.append({
                "title": c["title"],
                "category": c["category"],
                "source": c["source"],
                "content": c["content"][:600],
                "score": float(scores[i]),
            })
        answer = self._synthesize(question, results)
        return {"question": question, "answer": answer, "sources": results}

    def _synthesize(self, question: str, sources: list[dict]) -> str:
        if not sources or sources[0]["score"] < 0.05:
            return (
                "I could not find a strong match in the project knowledge base. "
                "Try asking about RRC handover, O-RAN RIC, multi-agent consensus, security threats, or digital twin."
            )
        top = sources[0]
        q = question.lower()
        intro = f"Based on **{top['source']}** ({top['category']}):\n\n"
        if "consensus" in q:
            return intro + "Consensus uses Byzantine Fault Tolerance with weighted trust voting. Decisions require >70% agreement, trust >0.8, and confidence >0.85. " + top["content"][:400]
        if "agent" in q or "mobility" in q:
            return intro + top["content"][:500]
        if "3gpp" in q or "rrc" in q or "rsrp" in q:
            return intro + top["content"][:500]
        if "security" in q or "attack" in q or "jamming" in q:
            return intro + top["content"][:500]
        if "digital twin" in q or "simulation" in q:
            return intro + "The digital twin simulates 48 cells across 12 gNBs with real-time KPI drift, attack injection, and agent mitigation. " + top["content"][:300]
        return intro + top["content"][:500]
