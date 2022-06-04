import pickle
import json
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
import os


def get_tokens(sentence):
    return tokenizer.encode(sentence, add_special_tokens=False)


if __name__ == '__main__':
    parser = ArgumentParser(description='Tokenize by Transformers')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--pretrained_model', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    pretrained_model = args.pretrained_model
    save_dir = 'data/{}'.format(dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    print('Dataset: {}, Pretrained Model: {}\n'.format(
            dataset, pretrained_model))

    for t in ['train', 'val', 'test']:
        file = '../../dataset/{}/post/{}.json'.format(args.dataset, t)
        with open(file, 'r') as f:
            pieces = json.load(f)

        pieces_tokens = [get_tokens(p['content']) for p in tqdm(pieces)]
        df = pd.DataFrame(
            {'tokens_num': [len(tokens) for tokens in pieces_tokens]})

        print('File: {}'.format(file))
        print('Posts: {}\nTokens num: {}\n'.format(len(df), df.describe()))

        # Export
        with open(os.path.join(save_dir, '{}.pkl'.format(t)), 'wb') as f:
            pickle.dump(pieces_tokens, f)
        df.describe().to_csv(os.path.join(save_dir, '{}.csv'.format(t)))
