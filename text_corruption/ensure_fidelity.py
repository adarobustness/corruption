# -*- coding: utf-8 -*-

"""Ensure fidelity after corruption."""

import logging
import torch
from torch.nn import CosineSimilarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')


def obtain_similarity(sentence1, sentence2):
    """Obtain similarity between two sentences."""
    embeddings = model.encode([sentence1, sentence2])
    similarity = CosineSimilarity(dim=1, eps=1e-6)
    sim = similarity(torch.tensor(embeddings[0]).unsqueeze(0), torch.tensor(embeddings[1]).unsqueeze(0))
    return abs(sim.item())

if __name__ == "__main__":
    print(obtain_similarity("The cat sat on the mat.", "The cat sat on the mat."))