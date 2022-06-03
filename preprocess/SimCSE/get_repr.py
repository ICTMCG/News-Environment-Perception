# -*- coding: utf-8 -*-

import os
import json
import pickle
from argparse import ArgumentParser
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig, BertModel


class SimCSE(nn.Module):
    def __init__(self, pretrained, pool_type="cls", dropout_prob=0.3):
        super().__init__()
        conf = BertConfig.from_pretrained(pretrained)
        conf.attention_probs_dropout_prob = dropout_prob
        conf.hidden_dropout_prob = dropout_prob
        self.encoder = BertModel.from_pretrained(pretrained, config=conf)
        assert pool_type in [
            "cls", "pooler"], "invalid pool_type: %s" % pool_type
        self.pool_type = pool_type

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        if self.pool_type == "cls":
            output = output.last_hidden_state[:, 0]
        elif self.pool_type == "pooler":
            output = output.pooler_output
        return output


class PTMEncode(object):
    def __init__(self,
                 fname,
                 PTM_path,
                 batch_size,
                 max_length,
                 device):
        self.fname = fname
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(PTM_path)
        model = SimCSE(pretrained=PTM_path).to(device)
        self.model = model
        self.model.eval()
        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None

    def encode_batch(self, texts):
        text_encs = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        input_ids = text_encs["input_ids"].to(self.device)
        attention_mask = text_encs["attention_mask"].to(self.device)
        token_type_ids = text_encs["token_type_ids"].to(self.device)
        with torch.no_grad():
            output = self.model.forward(
                input_ids, attention_mask, token_type_ids)
        return output

    def encode_news(self, L2_normalize=False):
        all_vecs = []
        data = json.load(open(self.fname, 'r'))
        file_len = len(data)
        content_lst = [d['content'] for d in data]
        num_batch = file_len // self.batch_size + 1
        last_batch_pad = num_batch * self.batch_size - file_len
        texts = []
        for idx, line in tqdm(enumerate(content_lst), total=file_len):
            if not line.strip():
                print("Empty content, whose index is {}.".format(idx))
                exit()
            texts.append(line.strip())
            if idx == file_len-1:
                # pad the last batch to batch_size
                for _ in range(last_batch_pad):
                    texts.append(line)
                assert len(texts) == self.batch_size
                vecs = self.encode_batch(texts)
                if L2_normalize:
                    vecs = vecs / vecs.norm(dim=1, keepdim=True)
                all_vecs.append(vecs.cpu())
                texts = []
            if len(texts) >= self.batch_size:
                vecs = self.encode_batch(texts)
                if L2_normalize:
                    vecs = vecs / vecs.norm(dim=1, keepdim=True)
                all_vecs.append(vecs.cpu())
                texts = []

        all_vecs = torch.cat(all_vecs, 0)
        all_vecs = all_vecs[:file_len, :]

        assert all_vecs.size()[0] == file_len

        return all_vecs

    def encode_post_a_dataset(self, datatype, L2_normalize=False):
        all_vecs = []
        data = json.load(
            open(os.path.join(self.fname, datatype + '.json'), 'r'))
        file_len = len(data)
        content_lst = [d['content'] for d in data]
        num_batch = file_len // self.batch_size + 1
        last_batch_pad = num_batch * self.batch_size - file_len
        texts = []
        for idx, line in tqdm(enumerate(content_lst), total=file_len):
            if not line.strip():
                print("Empty content, whose index is {}.".format(idx))
                exit()
            texts.append(line.strip())
            if idx == file_len-1:
                # pad the last batch to batch_size
                for _ in range(last_batch_pad):
                    texts.append(line)
                assert len(texts) == self.batch_size
                vecs = self.encode_batch(texts)
                if L2_normalize:
                    vecs = vecs / vecs.norm(dim=1, keepdim=True)
                all_vecs.append(vecs.cpu())
                texts = []
            if len(texts) >= self.batch_size:
                vecs = self.encode_batch(texts)
                if L2_normalize:
                    vecs = vecs / vecs.norm(dim=1, keepdim=True)
                all_vecs.append(vecs.cpu())
                texts = []

        all_vecs = torch.cat(all_vecs, 0)
        all_vecs = all_vecs[:file_len, :]

        assert all_vecs.size()[0] == file_len

        print('Finish encoding {}.json!'.format(datatype))

        return all_vecs

    def encode_post(self):
        train_vecs = self.encode_post_a_dataset(datatype='train')
        val_vecs = self.encode_post_a_dataset(datatype='val')
        test_vecs = self.encode_post_a_dataset(datatype='test')

        return train_vecs, val_vecs, test_vecs


if __name__ == "__main__":
    parser = ArgumentParser(description='get_repr')
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dataset = args.dataset
    SimCSE_ckpt = './train_SimCSE/ckpts/{}'.format(dataset)
    if dataset == 'Chinese':
        MAX_LEN = 256
    elif dataset == 'English':
        MAX_LEN = 128

    # --------------------- for news --------------------- #
    simcse = PTMEncode(
        fname='../../dataset/{}/news/news.json'.format(dataset),
        PTM_path=SimCSE_ckpt,
        batch_size=256,
        max_length=MAX_LEN,
        device=device)
    all_vecs = simcse.encode_news()
    # save all_vecs
    with open('./data/{}/news/all_vecs_origin.pkl'.format(dataset), 'wb') as f:
        pickle.dump(all_vecs, f)

    # --------------------- for post --------------------- #
    simcse = PTMEncode(
        fname='../../dataset/{}/post'.format(dataset),
        PTM_path=SimCSE_ckpt,
        batch_size=256,
        max_length=MAX_LEN,
        device=device)
    train_vecs, val_vecs, test_vecs = simcse.encode_post()
    for d in ['train', 'val', 'test']:
        with open('./data/{}/post/{}_vecs_origin.pkl'.format(dataset, d), 'wb') as f:
            pickle.dump(eval(d+'_vecs'), f)
