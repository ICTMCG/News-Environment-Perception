from gensim.summarization import bm25
import numpy as np
import json
from tqdm import tqdm


def get_total_scores(queries, corpus, bm25_model=None):
    # queries, corpus: texts' list
    
    if not bm25_model:
        print('Init BM25 model...')
        bm25_model = init_model(corpus)
        
    bm25_scores = np.zeros((len(queries), len(corpus)))

    for i, search in enumerate(tqdm(queries)):
        scores = np.array(bm25_model.get_scores(search))
        bm25_scores[i] = scores

    print('bm25_scores: ', bm25_scores.shape)
    return bm25_scores


def init_model(corpus):
    return bm25.BM25(corpus)


def get_a_query_scores(query, bm25_model):
    return np.array(bm25_model.get_scores(query))
