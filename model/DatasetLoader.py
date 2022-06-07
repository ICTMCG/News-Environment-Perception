import pickle
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import json
import time
from tqdm import tqdm
from config import INDEX_OF_LABEL, kernel_mu, kernel_sigma, ZERO


class DatasetLoader(Dataset):
    def __init__(self, args, split_set):
        print('\n', '=' * 20, '\n')
        self.args = args
        self.split_set = split_set

        # Fake News Detectors
        pieces_file = '../dataset/{}/post/{}.json'.format(
            self.args.dataset, split_set)
        with open(pieces_file, 'r') as f:
            pieces = json.load(f)

        if args.model == 'BERT':
            tokens_file = '../preprocess/BERT/data/{}/{}.pkl'.format(
                self.args.dataset, split_set)

            # Emotion Features
            if args.bert_use_emotion:
                emotions_file = '../preprocess/Emotion/data/{}/emotions/{}.npy'.format(
                    self.args.dataset, split_set)
                self.emotion_features = np.load(emotions_file)
                self.emotion_features = self.tensorize(self.emotion_features)
                print('Loading Emotion Features:\n File: {}, Shape: {}\n'.format(
                    emotions_file, self.emotion_features.shape))

        else:
            tokens_file = '../preprocess/WordEmbeddings/data/{}/{}.pkl'.format(
                self.args.dataset, split_set)

            # Event Labels
            if args.model == 'EANN':
                event_file = '../preprocess/EANN/data/{}/{}_event_labels.npy'.format(
                    self.args.dataset, split_set)
                self.event_labels = np.load(event_file)
                self.event_labels = self.tensorize(
                    self.event_labels, dt=torch.long)
                print('Loading Event Labels:\n File: {}, Shape: {}\n'.format(
                    event_file, self.event_labels.shape))

        with open(tokens_file, 'rb') as f:
            tokens = pickle.load(f)

        print('\n{}\nLoading: {}, \n{}\n{}\n'.format(
            '-'*10, pieces_file, tokens_file, '-'*10))

        self.labels = [INDEX_OF_LABEL[p['label']] for p in pieces]
        self.tokens = tokens

        if args.model in ['DeClarE', 'MAC']:
            print('\nInit Articles...')
            t = time.time()
            self.init_articles()
            print('\nDone, it took {:.2f}s\n'.format(time.time() - t))

        if not args.use_news_env:
            print('Do not use News Env Features...')
            print('\n', '=' * 20, '\n')
            return

        # Configuration
        self.macro_env_days = args.macro_env_days  # eg: 3
        self.micro_env_rate = args.micro_env_rate  # eg: 0.1
        self.micro_env_min_num = args.micro_env_min_num  # eg: 10
        print('Configuration: \nMacro Env: {} days, Micro Env: Top {:.2%} of macro env.\n'.format(
            self.macro_env_days, self.micro_env_rate))

        # News Env Init
        print('\nInit News Env...')
        t = time.time()

        tmp_data_dir = '../preprocess/NewsEnv/data/{}/_tmp/'.format(
            self.args.dataset)
        if not os.path.exists(tmp_data_dir):
            os.mkdir(tmp_data_dir)

        tmp_data_file = '{}_{}d_{}rate.pkl'.format(
            split_set, self.macro_env_days, self.micro_env_rate)
        tmp_data_file = os.path.join(tmp_data_dir, tmp_data_file)
        self.news_env_init(tmp_data_file)

        print('Done, it took {:.2f}s\n'.format(time.time() - t))

        print('\n', '=' * 20, '\n')

    def news_env_init(self, tmp_data_file):
        """
        # SimCSERepr:
        #    {'post': (#posts, 768),
        #     'news': (#news, 768)} (PS: del 'news' key to save memory)
        # MacroEnvAvg:
        #   {post_idx_1: avg_mac_vec_1, post_idx_2: avg_mac_vec_2, ...}
        # MicroEnvAvg:
        #   {post_idx_1: avg_mic_vec_1, post_idx_2: avg_mic_vec_2, ...}
        # SimDict:
        #   {'p-mac': {post_idx_1: [sim1, sim2, ...], post_idx_2: [sim1, sim2, ...]},
        #    'p-mic': {post_idx_1: [sim1, sim2, ...], post_idx_2: [sim1, sim2, ...]},
        #    'avgmic-mic': {post_idx_1: [sim1, sim2, ...], post_idx_2: [sim1, sim2, ...]}
        # KernelFeatures:
        #   {'p-mac': {post_idx_1: features_1, post_idx_2: features_2, ...},
        #    'p-mic': {post_idx_1: features_1, post_idx_2: features_2, ...},
        #    'avgmic-mic': {post_idx_1: features_1, post_idx_2: features_2, ...}}
        """

        if os.path.exists(tmp_data_file):
            with open(tmp_data_file, 'rb') as f:
                tmp_dict = pickle.load(f)
            self.SimCSERepr = tmp_dict['SimCSERepr']
            self.MacroEnvAvg = tmp_dict['MacroEnvAvg']
            self.MicroEnvAvg = tmp_dict['MicroEnvAvg']
            self.SimDict = tmp_dict['SimDict']
            self.KernelFeatures = tmp_dict['KernelFeatures']
            return

        dataset = self.args.dataset
        split_set = self.split_set

        # SimCSE Representations
        post_vecs_file = '../preprocess/SimCSE/data/{}/post/{}_vecs_origin.pkl'.format(
            dataset, split_set)
        news_vecs_file = '../preprocess/SimCSE/data/{}/news/all_vecs_origin.pkl'.format(
            dataset)
        with open(post_vecs_file, 'rb') as f:
            post_vecs = pickle.load(f)
        with open(news_vecs_file, 'rb') as f:
            news_vecs = pickle.load(f)
        self.SimCSERepr = {'post': post_vecs}
        print('SimCSE Repr: \nPost: {}, {}\nNews: {}, {}\n'.format(
            post_vecs_file, post_vecs.shape, news_vecs_file, news_vecs.shape))

        # Calculation
        post_env_file = '../preprocess/NewsEnv/data/{}/{}_data_{}d.pkl'.format(
            dataset, split_set, self.macro_env_days)
        with open(post_env_file, 'rb') as f:
            post_env_pairs = pickle.load(f)

        print('\n Calculating MacroEnvAvg, MicroEnvAvg, SimDict, and KernelFeatures ...')
        self.MacroEnvAvg = dict()
        self.MicroEnvAvg = dict()
        self.SimDict = {'p-mac': dict(), 'p-mic': dict(), 'avgmic-mic': dict()}
        self.KernelFeatures = {
            'p-mac': dict(), 'p-mic': dict(), 'avgmic-mic': dict()}

        cos_func = torch.nn.CosineSimilarity(dim=1)
        self.kernel_mu = self.tensorize(kernel_mu)
        self.kernel_sigma = self.tensorize(kernel_sigma)

        for i, macro_env_pairs in enumerate(tqdm(post_env_pairs)):
            macro_env_idxs = [p[0] for p in macro_env_pairs]
            micro_env_num = int(max(self.micro_env_min_num, len(
                macro_env_idxs) * self.micro_env_rate))

            if len(macro_env_pairs) == 0:
                print('\nPost {} 没有外部新闻!\n'.format(i))
                macro_env_vecs = torch.zeros_like(news_vecs[:micro_env_num])
            else:
                macro_env_vecs = news_vecs[macro_env_idxs]
            micro_env_vecs = macro_env_vecs[:micro_env_num]

            avg_macro_vec = torch.mean(macro_env_vecs, dim=0)
            avg_micro_vec = torch.mean(micro_env_vecs, dim=0)
            self.MacroEnvAvg[i] = avg_macro_vec
            self.MicroEnvAvg[i] = avg_micro_vec

            self.SimDict['p-mac'][i] = [p[1] for p in macro_env_pairs]
            self.SimDict['p-mic'][i] = self.SimDict['p-mac'][i][:micro_env_num]
            # (768) -> (micro_env_num, 768) -> (micro_env_num)
            try:
                self.SimDict['avgmic-mic'][i] = cos_func(
                    avg_micro_vec.repeat(micro_env_num, 1), micro_env_vecs).tolist()
            except:
                print('Idx: {}, len(macro_env_pairs) = {}'.format(i, len(macro_env_pairs)))
                print('micro_env_vecs: ', micro_env_vecs.shape)
                print('avg_micro_vec: ', avg_micro_vec.shape)
                exit()

            for k in ['p-mac', 'p-mic', 'avgmic-mic']:
                self.KernelFeatures[k][i] = self.gaussian_kernel_pooling(
                    self.SimDict[k][i]).tolist()

        # Export
        tmp_dict = {'SimCSERepr': self.SimCSERepr, 'MacroEnvAvg': self.MacroEnvAvg,
                    'MicroEnvAvg': self.MicroEnvAvg, 'SimDict': self.SimDict, 'KernelFeatures': self.KernelFeatures}
        with open(tmp_data_file, 'wb') as f:
            pickle.dump(tmp_dict, f)

    def gaussian_kernel_pooling(self, sim_values):
        k, n = len(kernel_mu), len(sim_values)

        if n == 0:
            return self.tensorize(torch.zeros(k))

        # (k) -> (n, k)
        mu = self.kernel_mu.repeat(n, 1)
        sigma = self.kernel_sigma.repeat(n, 1)

        # (n) -> (k, n) -> (n, k)
        sim_values = self.tensorize(sim_values)
        sim_values = sim_values.repeat(k, 1).T

        # (n, k) -> (k)
        kernel_features = torch.exp(-0.5 * ((sim_values - mu) * sigma)**2)
        kernel_features = torch.sum(kernel_features, dim=0)

        return kernel_features

    def init_articles(self):
        doc_num = self.args.relevant_articles_num

        article_tokens_file = '../preprocess/WordEmbeddings/data/{}/article/article.pkl'.format(
            self.args.dataset)
        with open(article_tokens_file, 'rb') as f:
            article_tokens = pickle.load(f)

        retreive_ideally = 'Ideal_' if self.args.retrieve_ideally else ''
        bm25_ranks_file = '../preprocess/BM25/data/{}/{}Top10_articles_{}.npy'.format(
            self.args.dataset, retreive_ideally, self.split_set)
        bm25_ranks = np.load(bm25_ranks_file)
        bm25_ranks = np.array(bm25_ranks, dtype=int)

        print('Article tokens: sz = {}, bm25_ranks.shape = {}'.format(
            len(article_tokens), bm25_ranks.shape))

        self.post_articles_tokens = dict()
        self.post_articles_masks = dict()
        for post_idx, ranks in enumerate(tqdm(bm25_ranks)):
            ranks = [r for r in ranks if r != -1]
            sz = min(doc_num, len(ranks))

            # tokens
            self.post_articles_tokens[post_idx] = [
                article_tokens[r] for r in ranks[:sz]]

            # mask: (#doc, 1)
            mask = torch.zeros(doc_num, 1, dtype=torch.float,
                               device=self.args.device)
            if sz != 0:
                mask[:-sz] = 1 / sz
            self.post_articles_masks[post_idx] = mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = (idx, self.labels[idx])
        return sample

    def tensorize(self, arr, dt=torch.float):
        if type(arr) == list and type(arr[0]) == torch.Tensor:
            arr = torch.stack(arr)

        return torch.as_tensor(arr, device=self.args.device, dtype=dt)
