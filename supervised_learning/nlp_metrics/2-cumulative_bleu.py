#!/usr/bin/env python3
"""
2-cumulative_bleu.py
"""
import numpy as np
ngram_bleu = __import__('1-ngram_bleu').ngram_bleu


def cumulative_bleu(references, sentence, n):
    """
    calculates the cumulative BLEU score for a sentence
    """
    bleu_scores = []
    for i in range(1, n + 1):
        bleu = ngram_bleu(references, sentence, i)
        bleu_scores.append(bleu)

    return np.prod(bleu_scores) ** (1 / n)
