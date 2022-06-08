import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.autograd import Variable, Function
from config import DATASETS_CHINESE, DATASETS_ENGLISH, MAX_TOKENS_OF_A_POST, ZERO
from transformers import BertModel
from NewsEnvExtraction import NewsEnvExtraction


class BiLSTM(nn.Module):
    def __init__(self, args):
        super(BiLSTM, self).__init__()

        self.args = args

        self.max_sequence_length = args.bilstm_input_max_sequence_length
        self.num_layers = args.bilstm_num_layer
        self.hidden_size = args.bilstm_hidden_dim

        weight = torch.load(
            '../preprocess/WordEmbeddings/data/{}/embedding_weight.pt'.format(args.dataset))
        weight.to(args.device)
        self.embedding = nn.Embedding.from_pretrained(weight)

        self.lstm = nn.LSTM(weight.shape[1], self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=args.bilstm_dropout)
        self.last_output = self.hidden_size * 2

    def forward(self, idxs, dataset):
        inputs = [self._encode(dataset.tokens[idx.item()]) for idx in idxs]

        # (bs, max_len)
        input_ids = torch.tensor(
            [i[0] for i in inputs], dtype=torch.long, device=self.args.device)
        # (bs, max_len, 1)
        masks = torch.stack([i[1] for i in inputs])

        # (bs, max_len, D)
        sentence = self.embedding(input_ids)
        # (bs, max_len, hidden_size*2)
        bilstm_out, _ = self.lstm(sentence)

        # (bs, hidden_size*2)
        semantic_output = torch.sum(masks*bilstm_out, dim=1)

        return semantic_output

    def _encode(self, doc):
        doc = doc[:self.max_sequence_length]

        padding_length = self.max_sequence_length - len(doc)
        input_ids = doc + [0] * padding_length

        mask = torch.zeros(self.max_sequence_length, 1, dtype=torch.float,
                           device=self.args.device)
        mask[:-padding_length] = 1 / len(doc)

        return input_ids, mask


class BERT(nn.Module):
    def __init__(self, args) -> None:
        super(BERT, self).__init__()

        self.args = args

        self.bert = BertModel.from_pretrained(
            args.bert_pretrained_model, return_dict=False)

        for name, param in self.bert.named_parameters():
            # finetune the pooler layer
            if name.startswith("pooler"):
                if 'bias' in name:
                    param.data.zero_()
                elif 'weight' in name:
                    param.data.normal_(
                        mean=0.0, std=self.bert.config.initializer_range)
                param.requires_grad = True

            # finetune the last encoder layer
            elif name.startswith('encoder.layer.11'):
                param.requires_grad = True

            # the embedding layer
            elif name.startswith('embeddings'):
                param.requires_grad = args.bert_training_embedding_layers

            # the other transformer layers (intermediate layers)
            else:
                param.requires_grad = args.bert_training_inter_layers

        fixed_layers = []
        for name, param in self.bert.named_parameters():
            if not param.requires_grad:
                fixed_layers.append(name)

        print('\n', '*'*15, '\n')
        print("BERT Fixed layers: {} / {}: \n{}".format(
            len(fixed_layers), len(self.bert.state_dict()), fixed_layers))
        print('\n', '*'*15, '\n')

        # [101, ..., 102, 103, 103, ..., 103]
        self.maxlen = args.bert_input_max_sequence_length
        self.doc_maxlen = self.maxlen - 2
        self.last_output = args.bert_hidden_dim

        if args.bert_use_emotion:
            self.mlp = nn.Linear(
                self.last_output + args.bert_emotion_features_dim, self.last_output)

    def forward(self, idxs, dataset):
        inputs = [self._encode(dataset.tokens[idx.item()]) for idx in idxs]

        # (batch_size, max_length)
        input_ids = torch.tensor(
            [i[0] for i in inputs], dtype=torch.long, device=self.args.device)
        # (batch_size, max_length, 1)
        masks = torch.stack([i[1] for i in inputs])

        # print('input_ids: ', input_ids.shape)
        # print('masks: ', masks.shape)

        # (batch_size, max_length, 768)
        seq_output, _ = self.bert(input_ids)

        # (batch_size, 768)
        semantic_output = torch.sum(masks*seq_output, dim=1)

        if self.args.bert_use_emotion:
            # (batch_size, 55)
            emotion_features = dataset.emotion_features[idxs]
            output = torch.cat([semantic_output, emotion_features], dim=-1)
            output = self.mlp(output)
        else:
            output = semantic_output

        return output

    def _encode(self, doc):
        doc = doc[:self.doc_maxlen]

        padding_length = self.maxlen - (len(doc) + 2)
        input_ids = [101] + doc + [102] + [103] * padding_length

        mask = torch.zeros(self.maxlen, 1, dtype=torch.float,
                           device=self.args.device)
        mask[:-padding_length] = 1 / (len(doc) + 2)

        return input_ids, mask


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()

        self.args = args

        self.input_dim = args.eann_input_dim
        self.hidden_dim = args.eann_hidden_dim  # 64
        self.max_sequence_length = args.eann_input_max_sequence_length

        weight = torch.load(
            '../preprocess/WordEmbeddings/data/{}/embedding_weight.pt'.format(args.dataset))
        weight.to(args.device)
        self.embedding = nn.Embedding.from_pretrained(weight)

        # TextCNN
        channel_in = 1
        filter_num = 20
        window_sizes = [1, 2, 3, 4]
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel_in, filter_num, (K, weight.shape[1])) for K in window_sizes])
        self.fc_cnn = nn.Linear(
            filter_num * len(window_sizes), self.hidden_dim)

        self.last_output = self.hidden_dim

    def forward(self, idxs, dataset):
        inputs = [self._encode(dataset.tokens[idx.item()]) for idx in idxs]

        # (bs, max_len)
        input_ids = torch.tensor(
            [i[0] for i in inputs], dtype=torch.long, device=self.args.device)
        # (bs, max_len, 1)
        masks = torch.stack([i[1] for i in inputs])

        # (bs, max_len, D)
        sentence = self.embedding(input_ids)
        sentence *= masks

        # [bs, 1, max_len, D]
        sentence = sentence.unsqueeze(1)
        # [bs, filter_num, feature_len] * len(window_sizes)
        sentence = [F.relu(conv(sentence)).squeeze(3) for conv in self.convs]
        # [bs, filter_num] * len(window_sizes)
        sentence = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in sentence]
        # [bs, filter_num * len(window_sizes)]
        sentence = torch.cat(sentence, 1)
        # [bs, hidden_dim]
        sentence = F.relu(self.fc_cnn(sentence))

        return sentence

    def _encode(self, doc):
        doc = doc[:self.max_sequence_length]

        padding_length = self.max_sequence_length - len(doc)
        input_ids = doc + [0] * padding_length

        mask = torch.zeros(self.max_sequence_length, 1, dtype=torch.float,
                           device=self.args.device)
        mask[:-padding_length] = 1 / len(doc)

        return input_ids, mask


class MLP(nn.Module):
    def __init__(self, args) -> None:
        super(MLP, self).__init__()

        assert args.use_news_env
        self.news_env_extractor = NewsEnvExtraction(args)

        # === MLP layers ===
        last_output = args.news_env_output_dim

        self.fcs = []
        for _ in range(args.num_mlp_layers - 1):
            curr_output = int(last_output / 2)
            self.fcs.append(nn.Linear(last_output, curr_output))
            last_output = curr_output
        self.fcs.append(nn.Linear(last_output, args.category_num))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, idxs, dataset):
        # (batch_size, news_env_dim)
        news_features = self.news_env_extractor(idxs, dataset)
        output = news_features

        for fc in self.fcs:
            output = F.gelu(fc(output))
        # (batch_size, category_num)
        return output


class EANN(nn.Module):
    ''' Learning Hierarchical Discourse-level Structures for Fake News Detection. NAACL 2019.'''
    """
        From https://github.com/yaqingwang/EANN-KDD18/blob/master/src/EANN_text.py
    """

    def __init__(self, args):
        super(EANN, self).__init__()
        self.args = args
        self.TextCNN = TextCNN(args)

        self.event_num = args.eann_event_num
        self.last_output = self.TextCNN.last_output

        self.event_discriminator = nn.Sequential(
            nn.Linear(self.last_output, self.last_output),
            nn.LeakyReLU(True),
            nn.Linear(self.last_output, self.event_num),
            nn.Softmax(dim=1)
        )

    def forward(self, idxs, dataset):
        # Fake News Detector: (bs, hidden_dim)
        detector_output = self.TextCNN(idxs, dataset)

        # Event Discrimination: (bs, hidden_dim) -> (bs, 300)
        reverse_text_feature = grad_reverse(detector_output)
        discriminator_output = self.event_discriminator(reverse_text_feature)

        return detector_output, discriminator_output


class GRL(Function):
    '''Gradient Reversal Layer'''
    """
        Refer to https://blog.csdn.net/t20134297/article/details/107870906
    """
    @staticmethod
    def forward(ctx, x, lamd):
        ctx.lamd = lamd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.lamd
        return grad_output, None


def grad_reverse(x, lamd=1):
    return GRL.apply(x, lamd)
