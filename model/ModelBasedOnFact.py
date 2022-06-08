import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from config import ZERO


class DeClarE(nn.Module):
    def __init__(self, args):
        super(DeClarE, self).__init__()

        self.args = args

        self.max_sequence_length = args.declare_input_max_sequence_lengtdh
        self.max_doc_len = args.declare_max_doc_length
        self.num_layers = args.declare_bilstm_num_layer
        self.D = args.declare_input_dim
        self.hidden_size = args.declare_hidden_dim

        weight = torch.load(
            '../preprocess/WordEmbeddings/data/{}/article/embedding_weight.pt'.format(args.dataset))
        weight.to(args.device)
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.D = weight.shape[1]

        self.Wa = nn.Linear(self.D * 2, 1)

        self.post_bilstm = nn.LSTM(self.D, self.hidden_size, self.num_layers,
                                   bidirectional=True, batch_first=True, dropout=args.declare_bilstm_dropout)
        self.article_bilstm = nn.LSTM(self.D, self.hidden_size, self.num_layers,
                                      bidirectional=True, batch_first=True, dropout=args.declare_bilstm_dropout)
        self.last_output = self.hidden_size * 4

    def forward(self, idxs, dataset):
        # ====== Input ======
        # --- Post ---
        post_inputs = [self._encode(dataset.tokens[idx.item()],
                                    max_len=self.max_sequence_length) for idx in idxs]

        # (bs, max_len)
        post_input_ids = torch.tensor(
            [i[0] for i in post_inputs], dtype=torch.long, device=self.args.device)
        # (bs, max_len, 1)
        post_masks = torch.stack([i[1] for i in post_inputs])
        # (bs, max_len, D)
        post = self.embedding(post_input_ids)

        # (bs, max_len, 2H)
        post_semantic, _ = self.post_bilstm(post)
        # (bs, 2H)
        post_semantic = torch.sum(post_masks * post_semantic, dim=1)

        # print('post: ', post, torch.any(torch.isnan(post)))

        # --- Articles ---
        article_inputs_ids = []
        article_masks = []
        for idx in idxs:
            inputs = [self._encode(doc, max_len=self.max_doc_len)
                      for doc in dataset.post_articles_tokens[idx.item()]]
            # (#doc, max_doc_len)
            input_ids = torch.tensor(
                [i[0] for i in inputs], dtype=torch.long, device=self.args.device)
            # (#doc, max_doc_len, 1)
            masks = torch.stack([i[1] for i in inputs])

            article_inputs_ids.append(input_ids)
            article_masks.append(masks)

        # (bs, #doc, max_doc_len)
        article_inputs_ids = torch.stack(article_inputs_ids)
        # (bs, #doc, max_doc_len, 1)
        article_masks = torch.stack(article_masks)

        # bs's (#doc, 1) -> (bs, #doc, 1)
        article_num_masks = [dataset.post_articles_masks[idx.item()]
                             for idx in idxs]
        article_num_masks = torch.stack(article_num_masks)

        # (bs, #doc, max_doc_len, D)
        articles = self.embedding(article_inputs_ids)
        # print('articles: ', articles, torch.any(torch.isnan(articles)))

        # ====== Claim Specific Attention ======
        # (bs, D)
        post = torch.sum(post_masks * post, dim=1)

        # (bs, #doc, max_doc_len, 2D)
        _, doc_num, max_doc_len, _ = articles.size()
        post_concat_articles = torch.cat(
            [articles, post[:, None, None, :].repeat(1, doc_num, max_doc_len, 1)], dim=-1)

        # print('post_concat_articles: ', post_concat_articles.shape, torch.any(torch.isnan(post_concat_articles)))

        # (bs, #doc, max_doc_len, 1)
        hd_prime = self.Wa(post_concat_articles)
        # print('hd_prime: ', hd_prime.shape, torch.any(torch.isnan(hd_prime)))
        # hd_prime = hd_prime.masked_fill((article_masks == 0) != 0, -np.inf)
        # hd_prime = hd_prime.masked_fill(
        #     (article_num_masks.unsqueeze(-2).repeat(1, 1, max_doc_len, 1) == 0) != 0, 0)
        # print('hd_prime: ', hd_prime.shape, torch.any(torch.isnan(hd_prime)))
        alphad = F.softmax(hd_prime, dim=2)

        # print('alphad: ', alphad, torch.any(torch.isnan(alphad)))

        # === Articles ===
        # (bs * #docs, max_doc_len, D)
        hd = articles.view(-1, max_doc_len, self.D)
        hd, _ = self.article_bilstm(hd)
        # (bs, #doc, max_doc_len, 2H)
        hd = hd.view(-1, doc_num, max_doc_len, 2 * self.hidden_size)

        # print('hd: ', hd.shape, torch.any(torch.isnan(hd)))

        # === Credibility Score ===
        # (bs, #doc, 2H)
        output = torch.sum(alphad * hd * article_masks, dim=-2)
        # (bs, 2H)
        output = torch.sum(output * article_num_masks, dim=-2)

        # (bs, 4H)
        output = torch.cat([post_semantic, output], dim=-1)

        # print('DeClarE output: ', output, torch.any(torch.isnan(output)))
        return output

    def _encode(self, doc, max_len):
        doc = doc[:max_len]

        padding_length = max_len - len(doc)
        input_ids = doc + [0] * padding_length

        mask = torch.zeros(max_len, 1, dtype=torch.float,
                           device=self.args.device)

        if len(doc) != 0:
            mask[:-padding_length] = 1 / len(doc)

        return input_ids, mask


class MAC(nn.Module):
    '''Hierarchical Multi-head Attentive Network for Evidence-aware Fake News Detection. EACL 2021.'''
    """
        Refer to https://github.com/nguyenvo09/EACL2021/blob/9d04d8954c1ded2110daac23117de11221f08cc6/Models/FCWithEvidences/hierachical_multihead_attention.py
    """

    def __init__(self, args):
        super(MAC, self).__init__()
        self.args = args

        self.max_sequence_length = args.mac_input_max_sequence_length
        self.max_doc_len = args.mac_max_doc_length
        self.input_dim = args.mac_input_dim  
        self.hidden_size = args.mac_hidden_dim 
        self.dropout_doc = args.mac_dropout_doc  
        self.dropout_query = args.mac_dropout_query
        self.num_heads_1 = args.mac_nhead_1
        self.num_heads_2 = args.mac_nhead_2
        self.num_layers = 1
        
        weight = torch.load(
            '../preprocess/WordEmbeddings/data/{}/article/embedding_weight.pt'.format(args.dataset))
        weight.to(args.device)
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.input_dim = weight.shape[1]

        self.doc_bilstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                                  bidirectional=True, batch_first=True, dropout=self.dropout_doc)
        self.query_bilstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                                    bidirectional=True, batch_first=True, dropout=self.dropout_query)

        self.W1 = nn.Linear(4 * self.hidden_size, 2 *
                            self.hidden_size, bias=False)
        self.W2 = nn.Linear(2 * self.hidden_size, self.num_heads_1, bias=False)
        self.W3 = nn.Linear((self.num_heads_1 + 1) * 2 *
                            self.hidden_size, 2 * self.hidden_size, bias=False)
        self.W4 = nn.Linear(2 * self.hidden_size, self.num_heads_2, bias=False)

        # self.query_features_dropout = nn.Dropout(self.dropout_query)
        # self.doc_features_dropout = nn.Dropout(self.dropout_doc)

        last_output_dim = 2 * self.hidden_size * \
            (1 + self.num_heads_1 * self.num_heads_2)
        self.W5 = nn.Linear(last_output_dim, 2 * self.hidden_size)
        self.last_output = 4 * self.hidden_size

    def forward(self, idxs, dataset):
        # ====== Input ======
        # --- Post ---
        post_inputs = [self._encode(dataset.tokens[idx.item()],
                                    max_len=self.max_sequence_length) for idx in idxs]

        # (bs, max_len)
        post_input_ids = torch.tensor(
            [i[0] for i in post_inputs], dtype=torch.long, device=self.args.device)
        # (bs, max_len, 1)
        post_masks = torch.stack([i[1] for i in post_inputs])
        # (bs, max_len, D)
        post = self.embedding(post_input_ids)

        # --- Articles ---
        article_inputs_ids = []
        article_masks = []
        for idx in idxs:
            inputs = [self._encode(doc, max_len=self.max_doc_len)
                      for doc in dataset.post_articles_tokens[idx.item()]]
            # (#doc, max_doc_len)
            input_ids = torch.tensor(
                [i[0] for i in inputs], dtype=torch.long, device=self.args.device)
            # (#doc, max_doc_len, 1)
            masks = torch.stack([i[1] for i in inputs])

            article_inputs_ids.append(input_ids)
            article_masks.append(masks)

        # (bs, #doc, max_doc_len)
        article_inputs_ids = torch.stack(article_inputs_ids)
        # (bs, #doc, max_doc_len, 1)
        article_masks = torch.stack(article_masks)

        # bs's (#doc, 1) -> (bs, #doc, 1)
        article_num_masks = [dataset.post_articles_masks[idx.item()]
                             for idx in idxs]
        article_num_masks = torch.stack(article_num_masks)

        # (bs, #doc, max_doc_len, D)
        articles = self.embedding(article_inputs_ids)
        # print('articles: ', articles, torch.any(torch.isnan(articles)))

        # ====== Forward ======

        # post: (bs, max_len, D)
        # post_masks: (bs, max_len, 1)
        # articles: (bs, #doc, max_doc_len, D)
        # article_masks: (bs, #doc, max_doc_len, 1)
        # article_num_masks: (bs, #doc, 1)

        # (bs, max_len, 2H)
        query_hiddens, _ = self.query_bilstm(post)
        # (bs, 2H)
        query_repr = torch.sum(query_hiddens * post_masks, dim=1)
        # query_repr = self.query_features_dropout(query_repr)

        # TimeDistributed(BiLSTM)
        df_sizes = articles.size()
        # (bs * #doc, max_doc_len, D)
        doc_hiddens = articles.view(-1, df_sizes[-2], df_sizes[-1])
        # (bs * #doc, max_doc_len, 2H)
        doc_hiddens, _ = self.doc_bilstm(doc_hiddens)
        # (bs, #doc, max_doc_len, 2H)
        doc_hiddens = doc_hiddens.view(
            df_sizes[0], df_sizes[1], df_sizes[2], doc_hiddens.size()[-1])

        # doc_hiddens = self.doc_features_dropout(doc_hiddens)

        # Multi-head Word Attention Layer
        C1 = query_repr.unsqueeze(1).unsqueeze(1).repeat(
            1, doc_hiddens.shape[1], doc_hiddens.shape[2], 1)  # [batch_size, #doc, doc_len, hidden_size * 2]
        # [batch_size, #doc, doc_len, hidden_size*4]
        A1 = torch.cat((doc_hiddens, C1), dim=-1)
        # [batch_size, #doc, doc_len, head_num_1]
        A1 = self.W2(torch.tanh(self.W1(A1)))

        # exclude the padding words in each doc
        A1 = F.softmax(A1, dim=-1)  # [batch_size, #doc, doc_len, head_num_1]
        A1 = A1.masked_fill((article_masks == 0), 0)

        # [batch_size * #doc, doc_len, head_num_1]
        A1_tmp = A1.reshape(-1, A1.shape[-2], A1.shape[-1])
        # [batch_size * #doc, doc_len, hidden_size * 2]
        doc_hiddens_tmp = doc_hiddens.reshape(-1,
                                              doc_hiddens.shape[-2], doc_hiddens.shape[-1])
        # [batch_size*#doc, head_num_1, doc_len] * [batch_size*#doc, doc_len, hidden_size * 2]
        D = torch.bmm(A1_tmp.permute(0, 2, 1), doc_hiddens_tmp)
        # [batch_size, #doc, head_num_1 * hidden_size * 2]
        D = D.view(A1.shape[0], A1.shape[1], -1)

        # Multi-head Document Attention Layer
        # [batch_size, #doc, hidden_size * 2]
        C2 = query_repr.unsqueeze(1).repeat(1, D.shape[1], 1)
        # [batch_size, #doc, (head_num_1 + 1) * hidden_size * 2]
        A2 = torch.cat((D, C2), dim=-1)
        A2 = self.W4(torch.tanh(self.W3(A2)))  # [batch_size, #doc, head_num_2]

        # [batch_size, #doc, 1]
        A2 = F.softmax(A2, dim=-1)  # [batch_size, #doc, head_num_2]
        A2 = A2.masked_fill((article_num_masks == 0), 0)

        # [batch_size, #doc, head_num_2] * [batch_size, #doc, head_num_1 * hidden_size * 2]
        D = torch.bmm(A2.permute(0, 2, 1), D)
        # [batch_size, head_num_2 * head_num_1 * hidden_size * 2] Eq.(9)
        D = D.view(D.shape[0], -1)

        # Output Layer
        # [batch_size, (head_num_2 * head_num_1 * 2 + 2) * hidden_size]
        output = torch.cat((query_repr, D), dim=-1)
        # (bs, 2H)
        output = self.W5(output)
        # (bs, 4H)
        output = torch.cat([query_repr, output], dim=-1)

        return output

    def _encode(self, doc, max_len):
        doc = doc[:max_len]

        padding_length = max_len - len(doc)
        input_ids = doc + [0] * padding_length

        mask = torch.zeros(max_len, 1, dtype=torch.float,
                           device=self.args.device)

        if len(doc) != 0:
            mask[:-padding_length] = 1 / len(doc)

        return input_ids, mask
