import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class KnowledgeBuffer:
    def __init__(self, model_name='xlm-roberta-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.eval()
        
        # In-memory "Verified Knowledge" store (Simulated for research)
        # In production, this would be a Vector DB like ChromaDB or Faiss
        self.knowledge_base = [
            "The World Health Organization confirms that tea does not cure COVID-19.",
            "NASA released official images of the Mars Rover on the red planet surface.",
            "The RBI announced a repo rate hike of 25 basis points recently.",
            "Government officials confirmed there is no free recharge scheme currently.",
            "Scientists have debunked the claim that 5G towers cause virus spread."
        ]
        
        # Pre-compute embeddings for the knowledge base
        self.knowledge_embeddings = self._encode_texts(self.knowledge_base)

    def _encode_texts(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use Mean Pooling for sentence embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def get_relevant_context(self, query_text):
        """
        Retrieves the most semantically similar verified fact from the buffer.
        """
        query_embedding = self._encode_texts([query_text])
        
        # Compute Cosine Similarity
        similarities = torch.nn.functional.cosine_similarity(query_embedding, self.knowledge_embeddings)
        best_idx = torch.argmax(similarities).item()
        
        return self.knowledge_embeddings[best_idx], self.knowledge_base[best_idx], similarities[best_idx].item()

if __name__ == "__main__":
    kb = KnowledgeBuffer()
    feat, text, score = kb.get_relevant_context("Is there a cure for COVID in tea?")
    print(f"Query: Is there a cure for COVID in tea?")
    print(f"Retrieved Context: {text} (Score: {score:.4f})")
