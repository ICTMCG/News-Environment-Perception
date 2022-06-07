from numpy.lib.shape_base import tile
import torch
from torch._C import device
import torch.nn as nn
from config import kernel_mu, kernel_sigma, ZERO
import time
import pickle
import torch.nn.functional as F


class NewsEnvExtraction(nn.Module):
    def __init__(self, args) -> None:
        super(NewsEnvExtraction, self).__init__()
        self.args = args

        self.kernel_mu = self.tensorize(kernel_mu)
        self.kernel_sigma = self.tensorize(kernel_sigma)

        self.macro_env_output_dim = args.macro_env_output_dim
        self.micro_env_output_dim = args.micro_env_output_dim
        # self.news_env_output_dim = args.news_env_output_dim

        macro_output = 0
        if self.args.use_semantics_of_news_env:
            macro_output += 2 * args.bert_hidden_dim
        if self.args.use_similarity_of_news_env:
            macro_output += len(kernel_mu)

        self.macro_mlp = nn.Linear(macro_output, self.macro_env_output_dim)

        micro_output = 0
        if self.args.use_semantics_of_news_env:
            self.micro_sem_mlp = nn.Linear(
                2 * args.bert_hidden_dim, self.micro_env_output_dim)
            micro_output += self.micro_env_output_dim
        if self.args.use_similarity_of_news_env:
            self.micro_sim_mlp = nn.Linear(
                2 * len(kernel_mu), self.micro_env_output_dim)
            micro_output += self.micro_env_output_dim

        self.micro_mlp = nn.Linear(micro_output, self.micro_env_output_dim)

    def forward(self, post_idxs, dataset):
        """
        post_idxs: the indexes of posts
        dataset: an instance of DatasetLoader in `DatasetLoader.py`
        """
        # ------------------ Macro Env ------------------
        if not self.args.use_semantics_of_news_env:
            p = None
            avg_emac = None
        else:
            # (bs, 768)
            p = [dataset.SimCSERepr['post'][idx.item()] for idx in post_idxs]
            p = self.tensorize(p)

            # (bs, 768)
            avg_emac = [dataset.MacroEnvAvg[idx.item()] for idx in post_idxs]
            avg_emac = self.tensorize(avg_emac)

        if not self.args.use_similarity_of_news_env:
            kernel_p_emac = None
        else:
            # p_emac_sims = [dataset.SimDict['p-mac'][idx.item()]
            #                for idx in post_idxs]
            # # (bs, #kernels)
            # kernel_p_emac = [self.gaussian_kernel_pooling(
            #     sims) for sims in p_emac_sims]
            # kernel_p_emac = self.tensorize(kernel_p_emac)

            kernel_p_emac = [dataset.KernelFeatures['p-mac'][idx.item()]
                             for idx in post_idxs]
            kernel_p_emac = self.normalize(self.tensorize(kernel_p_emac))

        vectors = [x for x in [p, avg_emac, kernel_p_emac] if x is not None]

        # (bs, macro_env_output_dim)
        v_p_mac = torch.cat(vectors, dim=-1)
        v_p_mac = self.macro_mlp(v_p_mac)

        # ------------------ Micro Env ------------------
        if not self.args.use_semantics_of_news_env:
            usem = None
        else:
            # (bs, 768)
            avg_emic = [dataset.MicroEnvAvg[idx.item()] for idx in post_idxs]
            avg_emic = self.tensorize(avg_emic)

            # (bs, 2 * 768) -> (bs, micro_env_output_dim)
            usem = torch.cat([p, avg_emic], dim=1)
            usem = self.micro_sem_mlp(usem)

        if not self.args.use_similarity_of_news_env:
            usim = None
        else:
            # p_emic_sims = [dataset.SimDict['p-mic'][idx.item()]
            #                for idx in post_idxs]
            # # (bs, #kernels)
            # kernel_p_emic = [self.gaussian_kernel_pooling(
            #     sims) for sims in p_emic_sims]
            # kernel_p_emic = self.tensorize(kernel_p_emic)

            kernel_p_emic = [dataset.KernelFeatures['p-mic'][idx.item()]
                             for idx in post_idxs]
            kernel_p_emic = self.normalize(self.tensorize(kernel_p_emic))

            # avgmic_emic_sims = [dataset.SimDict['avgmic-mic'][idx.item()]
            #                     for idx in post_idxs]
            # # (bs, #kernels)
            # kernel_avgmic_emic = [self.gaussian_kernel_pooling(
            #     sims) for sims in avgmic_emic_sims]
            # kernel_avgmic_emic = self.tensorize(kernel_avgmic_emic)

            kernel_avgmic_emic = [dataset.KernelFeatures['avgmic-mic'][idx.item()]
                                  for idx in post_idxs]
            kernel_avgmic_emic = self.normalize(
                self.tensorize(kernel_avgmic_emic))

            # Comparision
            # (bs, 2 * #kernels) -> (bs, micro_env_output_dim)
            usim = torch.cat([kernel_p_emic * kernel_avgmic_emic,
                              kernel_p_emic - kernel_avgmic_emic], dim=1)
            # print('usim: ', usim.shape, usim)
            usim = self.micro_sim_mlp(usim)

        vectors = [x for x in [usim, usem] if x is not None]

        # (bs, micro_env_output_dim)
        v_p_mic = torch.cat(vectors, dim=-1)
        # print('v_p_mic: ', v_p_mic.shape, v_p_mic)
        v_p_mic = self.micro_mlp(v_p_mic)
        # print('after micro_mlp, v_p_mic: ', v_p_mic)

        # # --------- Interaction (eg: Multi-head Attention) ---------

        # # Strategy1: Concat
        # out = torch.cat([v_p_mac, v_p_mic], dim=1)
        # out = self.mlp(out)

        # --------- In-batch Learning ---------
        h_mac = None
        h_mic = None

        if not self.args.use_macro_env:
            v_p_mac = torch.zeros_like(v_p_mac, device=self.args.device)
        if not self.args.use_micro_env:
            v_p_mic = torch.zeros_like(v_p_mic, device=self.args.device)

        return v_p_mac, v_p_mic, h_mac, h_mic

    def gaussian_kernel_pooling(self, sim_values):
        k, n = len(kernel_mu), len(sim_values)

        # (k) -> (n, k)
        mu = self.kernel_mu.repeat(n, 1)
        sigma = self.kernel_sigma.repeat(n, 1)

        # (n) -> (k, n) -> (n, k)
        sim_values = self.tensorize(sim_values)
        sim_values = sim_values.repeat(k, 1).T

        # (n, k) -> (k)
        kernel_features = torch.exp(-0.5 * ((sim_values - mu) * sigma)**2)
        kernel_features = torch.sum(kernel_features, dim=0)

        # # Scale
        # m, M = min(kernel_features), max(kernel_features)
        # kernel_features = (kernel_features - m) / (M - m + ZERO)

        return self.normalize(kernel_features)

    def normalize(self, kernel_features):
        # Normalize
        kernel_sum = torch.sum(kernel_features)
        kernel_features /= (kernel_sum + ZERO)

        return kernel_features

    def tensorize(self, arr, dt=torch.float):
        if type(arr) == list and type(arr[0]) == torch.Tensor:
            arr = torch.stack(arr)

        return torch.as_tensor(arr, device=self.args.device, dtype=dt)
