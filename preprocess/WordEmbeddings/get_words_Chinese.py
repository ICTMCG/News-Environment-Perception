import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
from argparse import ArgumentParser
import jieba
import os
import pickle


def load_embeddings(embeddings_file):
    """
    word embeddings
    """

    embeddings_index = {}
    with open(embeddings_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

        lines = lines if dataset == 'English' else lines[1:]

        for line in lines:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('File: {}, there are {} vectors'.format(
        embeddings_file, len(embeddings_index)))
    return embeddings_index


def get_words_tokens(words):
    # <unkown>: 0
    return [word2idx.get(w, 0) for w in words]


if __name__ == '__main__':
    parser = ArgumentParser(description='Tokenize by Jieba')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_type', type=str,
                        default='post', help=['post', 'article'])
    parser.add_argument('--embedding_file', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    embedding_file = args.embedding_file
    data_type = args.data_type

    save_dir = 'data/{}'.format(dataset)
    if data_type != 'post':
        save_dir = os.path.join(save_dir, data_type)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Dataset: {}, Data type: {}, Embedding_file: {}\n'.format(
        dataset, data_type, embedding_file))

    print('Loadding Embedding...')
    embedding_dict = load_embeddings(embeddings_file=embedding_file)
    word2idx = {k: i+1 for i, k in enumerate(embedding_dict)}
    idx2word = {v: k for k, v in word2idx.items()}
    D = len(embedding_dict[idx2word[1]])

    # (#words + 1, D): <padding>: 0, <unkown>: 0
    embedding_weight = torch.zeros(1 + len(word2idx), D, dtype=torch.float)
    for idx, word in idx2word.items():
        embedding_weight[idx] = torch.as_tensor(embedding_dict[word])

    torch.save(embedding_weight, os.path.join(save_dir, 'embedding_weight.pt'))
    del embedding_dict, embedding_weight
    print('Done.\n')

    if data_type == 'post':
        for t in ['train', 'val', 'test']:
            file = '../../dataset/{}/post/{}.json'.format(args.dataset, t)
            with open(file, 'r') as f:
                pieces = json.load(f)

            pieces_words = [list(jieba.cut(p['content']))
                            for p in tqdm(pieces)]
            pieces_tokens = [get_words_tokens(words)
                             for words in tqdm(pieces_words)]

            df = pd.DataFrame(
                {'tokens_num': [len(tokens) for tokens in pieces_tokens],
                 'unknown_tokens_num': [len([t for t in tokens if t == 0]) for tokens in pieces_tokens]})

            print('File: {}'.format(file))
            print('Posts: {}\nTokens num: {}\nUnknown Tokens num: {}\n'.format(
                len(df), df['tokens_num'].describe(), df['unknown_tokens_num'].describe()))

            # Export
            with open(os.path.join(save_dir, '{}.pkl'.format(t)), 'wb') as f:
                pickle.dump(pieces_tokens, f)
            df.describe().to_csv(os.path.join(save_dir, '{}.csv'.format(t)))

    elif data_type == 'article':
        t = data_type
        file = '../../dataset/{}/articles/articles.json'.format(
            args.dataset)
        with open(file, 'r') as f:
            pieces = json.load(f)

        for p in pieces:
            p['content'] = ''.join(p['content_all'])

        pieces_words = [list(jieba.cut(p['content'])) for p in tqdm(pieces)]
        pieces_tokens = [get_words_tokens(words)
                         for words in tqdm(pieces_words)]

        df = pd.DataFrame(
            {'tokens_num': [len(tokens) for tokens in pieces_tokens],
             'unknown_tokens_num': [len([t for t in tokens if t == 0]) for tokens in pieces_tokens]})

        print('File: {}'.format(file))
        print('Posts: {}\nTokens num: {}\nUnknown Tokens num: {}\n'.format(
            len(df), df['tokens_num'].describe(), df['unknown_tokens_num'].describe()))

        # Export
        with open(os.path.join(save_dir, '{}.pkl'.format(t)), 'wb') as f:
            pickle.dump(pieces_tokens, f)
        df.describe().to_csv(os.path.join(save_dir, '{}.csv'.format(t)))
