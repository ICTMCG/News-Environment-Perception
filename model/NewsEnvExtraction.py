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

        if self.args.in_batch_learning:
            self.macro_proj_mlp = nn.Linear(self.macro_env_output_dim, 1)
            self.micro_proj_mlp = nn.Linear(self.micro_env_output_dim, 1)

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
            kernel_avgmic_emic = self.normalize(self.tensorize(kernel_avgmic_emic))
            
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
        if self.args.in_batch_learning:
            # (bs, 1) -> (bs)
            h_mac = self.macro_proj_mlp(v_p_mac).squeeze()
            h_mic = self.micro_proj_mlp(v_p_mic).squeeze()

            h_mac = torch.sigmoid(h_mac)
            h_mic = torch.sigmoid(h_mic)
        else:
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


class SimValueFeatures(nn.Module):
    def __init__(self, args) -> None:
        super(SimValueFeatures, self).__init__()
        self.args = args

        # 统计sim_values的最大长度
        self.max_macro_num = 0
        self.max_micro_num = 0

        for split_set in ['train', 'val', 'test']:
            tmp_data_file = '../preprocess/SimCSE/data/{}/_tmp/{}_{}d_{}rate.pkl'.format(
                args.dataset, split_set, args.macro_env_days, args.micro_env_rate)
            with open(tmp_data_file, 'rb') as f:
                tmp_dict = pickle.load(f)
            sim_dict = tmp_dict['SimDict']

            macro_num = max([len(vals) for vals in sim_dict['p-mac'].values()])
            micro_num = max([len(vals) for vals in sim_dict['p-mic'].values()])

            self.max_macro_num = max(macro_num, self.max_macro_num)
            self.max_micro_num = max(micro_num, self.max_micro_num)

        print('\n最大新闻数量：macro_num={}，micro_num={}\n'.format(
            self.max_macro_num, self.max_micro_num))

        last_output = 0
        d = args.sim_values_output_dim
        if args.use_p_mac:
            self.mlp_p_mac = nn.Linear(self.max_macro_num, d)
            last_output += d
        if args.use_p_mic:
            self.mlp_p_mic = nn.Linear(self.max_micro_num, d)
            last_output += d
        if args.use_avgmic_mic:
            self.mlp_avgmic_mic = nn.Linear(self.max_micro_num, d)
            last_output += d

        # === MLP layers ===
        self.fcs = []
        for _ in range(args.num_mlp_layers - 1):
            curr_output = int(last_output / 2)
            self.fcs.append(nn.Linear(last_output, curr_output))
            last_output = curr_output
        self.fcs.append(nn.Linear(last_output, args.category_num))
        self.fcs = nn.ModuleList(self.fcs)

        # # 权值初始化
        # self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print('原始参数：', m.weight.data)
                nn.init.kaiming_normal_(m.weight.data)
                # print('初始化后的参数：', m.weight.data)

    def forward(self, post_idxs, dataset):
        if self.args.use_p_mac:
            p_emac_sims = [dataset.SimDict['p-mac'][idx.item()]
                           for idx in post_idxs]
            p_emac_sims = [self.padding(
                sims, max_num=self.max_macro_num) for sims in p_emac_sims]
            # (bs, max_macro_num)
            p_emac_sims = self.tensorize(p_emac_sims)

            v_p_mac = self.mlp_p_mac(p_emac_sims)
        else:
            v_p_mac = None

        if self.args.use_p_mic:
            p_emic_sims = [dataset.SimDict['p-mic'][idx.item()]
                           for idx in post_idxs]
            p_emic_sims = [self.padding(
                sims, max_num=self.max_micro_num) for sims in p_emic_sims]
            # (bs, max_micro_num)
            p_emic_sims = self.tensorize(p_emic_sims)

            print('p_mic_sims: ', p_emic_sims.shape, p_emic_sims)
            v_p_mic = self.mlp_p_mic(p_emic_sims)
            print('after mlp, v_p_mic: ', v_p_mic.shape, v_p_mic)

        else:
            v_p_mic = None

        if self.args.use_avgmic_mic:
            avgmic_emic_sims = [dataset.SimDict['avgmic-mic'][idx.item()]
                                for idx in post_idxs]
            avgmic_emic_sims = [self.padding(
                sims, max_num=self.max_micro_num) for sims in avgmic_emic_sims]
            # (bs, max_micro_num)
            avgmic_emic_sims = self.tensorize(avgmic_emic_sims)

            v_avgmic_mic = self.mlp_avgmic_mic(avgmic_emic_sims)
        else:
            v_avgmic_mic = None

        vectors = [v for v in [v_p_mac, v_p_mic,
                               v_avgmic_mic] if v is not None]
        output = torch.cat(vectors, dim=-1)

        for fc in self.fcs:
            output = F.gelu(fc(output))
            # output = fc(output)
        # (batch_size, category_num)
        return output

    def padding(self, sims, max_num, padding_value=0):
        padding_num = max_num - len(sims)
        return sims + [padding_value for _ in range(padding_num)]

    def tensorize(self, arr, dt=torch.float):
        if type(arr) == list and type(arr[0]) == torch.Tensor:
            arr = torch.stack(arr)

        return torch.as_tensor(arr, device=self.args.device, dtype=dt)


class SimValueKernelFeatures(nn.Module):
    def __init__(self, args) -> None:
        super(SimValueKernelFeatures, self).__init__()
        self.args = args

        kernels_dim = len(kernel_mu)

        last_output = 0
        d = args.sim_values_output_dim
        if args.use_p_mac:
            self.mlp_p_mac = nn.Linear(kernels_dim, d)
            last_output += d
        if args.use_p_mic:
            self.mlp_p_mic = nn.Linear(kernels_dim, d)
            last_output += d
        if args.use_avgmic_mic:
            self.mlp_avgmic_mic = nn.Linear(kernels_dim, d)
            last_output += d

        # === MLP layers ===
        self.fcs = []
        for _ in range(args.num_mlp_layers - 1):
            curr_output = int(last_output / 2)
            self.fcs.append(nn.Linear(last_output, curr_output))
            last_output = curr_output
        self.fcs.append(nn.Linear(last_output, args.category_num))
        self.fcs = nn.ModuleList(self.fcs)

        # # 权值初始化
        # self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print('原始参数：', m.weight.data)
                nn.init.kaiming_normal_(m.weight.data)
                # print('初始化后的参数：', m.weight.data)

    def forward(self, post_idxs, dataset):
        if self.args.use_p_mac:
            p_emac_sims = [dataset.KernelFeatures['p-mac'][idx.item()]
                           for idx in post_idxs]
            # (bs, kernels_num)
            p_emac_sims = self.tensorize(p_emac_sims)

            v_p_mac = self.mlp_p_mac(p_emac_sims)
        else:
            v_p_mac = None

        if self.args.use_p_mic:
            p_emic_sims = [dataset.KernelFeatures['p-mic'][idx.item()]
                           for idx in post_idxs]
            # (bs, kernels_num)
            p_emic_sims = self.tensorize(p_emic_sims)

            print('p_mic_sims: ', p_emic_sims.shape, p_emic_sims)
            v_p_mic = self.mlp_p_mic(p_emic_sims)
            print('after mlp, v_p_mic: ', v_p_mic.shape, v_p_mic)

        else:
            v_p_mic = None

        if self.args.use_avgmic_mic:
            avgmic_emic_sims = [dataset.KernelFeatures['avgmic-mic'][idx.item()]
                                for idx in post_idxs]
            # (bs, kernels_num)
            avgmic_emic_sims = self.tensorize(avgmic_emic_sims)

            v_avgmic_mic = self.mlp_avgmic_mic(avgmic_emic_sims)
        else:
            v_avgmic_mic = None

        vectors = [v for v in [v_p_mac, v_p_mic,
                               v_avgmic_mic] if v is not None]
        output = torch.cat(vectors, dim=-1)

        for fc in self.fcs:

            output = F.gelu(fc(output))
            # output = fc(output)

        print('output: ', output)
        print('=' * 10)
        return output

    def tensorize(self, arr, dt=torch.float):
        if type(arr) == list and type(arr[0]) == torch.Tensor:
            arr = torch.stack(arr)

        return torch.as_tensor(arr, device=self.args.device, dtype=dt)
