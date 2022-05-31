import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from NewsEnvExtraction import NewsEnvExtraction
from ModelBasedOnPattern import BERT, BiLSTM, TextCNN, EANN
from ModelBasedOnFact import DeClarE, MAC


class EnvEnhancedFramework(nn.Module):
    def __init__(self, args):
        super(EnvEnhancedFramework, self).__init__()

        self.args = args

        # === FEND ===
        if args.use_fake_news_detector:
            self.fake_news_detector = eval('{}(args)'.format(args.model))
            last_output = self.fake_news_detector.last_output
        else:
            last_output = 0

        # === Env ===
        if args.use_news_env:
            self.news_env_extractor = NewsEnvExtraction(args)

            if self.args.strategy_of_fusion == 'att':
                self.macro_multihead_attn = nn.MultiheadAttention(
                    args.multi_attention_dim, num_heads=8, dropout=0.5)
                self.micro_multihead_attn = nn.MultiheadAttention(
                    args.multi_attention_dim, num_heads=8, dropout=0.5)
            elif self.args.strategy_of_fusion == 'gate':
                assert args.macro_env_output_dim == args.micro_env_output_dim
                self.W_gate = nn.Linear(
                    self.fake_news_detector.last_output + args.macro_env_output_dim, args.macro_env_output_dim)

        # === MLP layers ===
        if args.use_news_env:
            if self.args.strategy_of_fusion == 'concat':
                last_output += args.macro_env_output_dim + args.micro_env_output_dim
            elif self.args.strategy_of_fusion == 'att':
                last_output += 2 * args.multi_attention_dim
            elif self.args.strategy_of_fusion == 'gate':
                last_output += args.macro_env_output_dim

        self.fcs = []
        for _ in range(args.num_mlp_layers - 1):
            curr_output = int(last_output / 2)
            self.fcs.append(nn.Linear(last_output, curr_output))
            last_output = curr_output
        self.fcs.append(nn.Linear(last_output, args.category_num))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, idxs, dataset):
        if not self.args.use_fake_news_detector:
            return self.forward_only_env(idxs, dataset)

        # (bs, FEND_last_output)
        detector_output = self.fake_news_detector(idxs, dataset)

        if self.args.model == 'EANN':
            detector_output, eann_event_output = detector_output

        if self.args.use_news_env:
            v_p_mac, v_p_mic, h_mac, h_mic = self.news_env_extractor(
                idxs, dataset)

            if self.args.strategy_of_fusion == 'concat':
                output = torch.cat([detector_output, v_p_mac, v_p_mic], dim=-1)
            elif self.args.strategy_of_fusion == 'att':
                # (1, bs, 768)
                key = detector_output.unsqueeze(0)
                value = key

                # env_output: (1, bs, emb_dim), env_weights: (bs, 1, 1)
                macro_output, macro_weights = self.macro_multihead_attn(
                    query=v_p_mac.unsqueeze(0), key=key, value=value)
                micro_output, micro_weights = self.micro_multihead_attn(
                    query=v_p_mic.unsqueeze(0), key=key, value=value)

                output = torch.cat(
                    [detector_output, macro_output.squeeze(0), micro_output.squeeze(0)], dim=-1)
            elif self.args.strategy_of_fusion == 'gate':
                # (bs, env_dim)
                g = torch.sigmoid(self.W_gate(
                    torch.cat([detector_output, v_p_mac], dim=-1)))
                # (bs, env_dim)
                v_p = g * v_p_mac + (1 - g) * v_p_mic
                # (bs, env_dim + FEND_last_output)
                output = torch.cat([detector_output, v_p], dim=-1)

        else:
            output = detector_output
            h_mac = None
            h_mic = None

        for fc in self.fcs:
            output = F.gelu(fc(output))

        if self.args.model == 'EANN':
            output = (output, eann_event_output)

        return output, h_mac, h_mic

    def forward_only_env(self, idxs, dataset):
        v_p_mac, v_p_mic, h_mac, h_mic = self.news_env_extractor(
            idxs, dataset)
        output = torch.cat([v_p_mac, v_p_mic], dim=-1)
        for fc in self.fcs:
            output = F.gelu(fc(output))
        return output, h_mac, h_mic

    def inference_analysis(self, idxs, dataset):
        # (bs, FEND_last_output)
        detector_output = self.fake_news_detector(idxs, dataset)

        assert self.args.use_news_env
        v_p_mac, v_p_mic, h_mac, h_mic = self.news_env_extractor(
            idxs, dataset)

        assert self.args.strategy_of_fusion == 'gate'

        # (bs, env_dim)
        g = torch.sigmoid(self.W_gate(
            torch.cat([detector_output, v_p_mac], dim=-1)))
        return g
