import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.records = None
        self.embeddings = None

    def fit(self, records):
        self.records = records
        texts = [i["text"] for i in records]
        embeddings = self.model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(embeddings)
        self.embeddings = embeddings
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query, top_k=3):
        query_embed = self.model.encode([query], convert_to_numpy=True).astype(
            "float32"
        )
        faiss.normalize_L2(query_embed)
        scores, indices = self.index.search(query_embed, top_k)
        res = []
        for score, idx in zip(scores[0], indices[0]):
            record = self.records[idx].copy()
            record["score"] = float(score)
            res.append(record)

        return res
