import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np
import sys
from data_loader import Dataset_ETT_hour


class Discretization(nn.Module):
    def __init__(self, num_bins=10, discr_type="one_hot"):
        super(Discretization, self).__init__()
        self.discr_type = discr_type
        self.num_bins = num_bins
        self.gap = -1

    def discretize(self, x):
        """
        x: [T,D]
        """
        self.gap = np.max(x, axis=0)[0] - np.min(x, axis=0)[0]
        x = x / self.gap
        x = np.floor((x * (self.num_bins - 1))).astype(int)
        return x

    def forward(self, x):
        return self.discretize(x)

    def decode(self, x):
        x = x / (self.num_bins - 1)
        x = x * self.gap
        return x


class BPEEncode_Channel_Splitting(nn.Module):
    def __init__(self):
        super(BPEEncode_Channel_Splitting, self).__init__()
        self.token_pair_max_list = []
        self.encode_list = []
        self.progress_max = 0
        self.word_dict_list = []
        self.max_min_code_list = []

    def progress_bar(self, x, inital=False, initial_max=0):
        if inital:
            self.progress_max = initial_max
        x = int((1 - (x / self.progress_max)) * 100)
        sys.stdout.write('\r')
        sys.stdout.write("Encode progress: {}%: ".format(x))
        sys.stdout.write("|")
        sys.stdout.write("▋" * x)
        sys.stdout.write(" " * int(100 - x))
        sys.stdout.write("|")
        sys.stdout.flush()

    def channel_split(self, x):
        return [x[:, ind] for ind in range(x.shape[1])]

    def find_max_token_pair(self, x, is_inital_progress=False):
        self.token_pair_max_list.clear()
        for channel in x:
            token_pair_dict = {}
            for i in range(len(channel) - 1):
                comp_0 = list(channel[i]) if isinstance(channel[i], tuple) else [channel[i]]
                comp_1 = list(channel[i + 1]) if isinstance(channel[i + 1], tuple) else [channel[i + 1]]
                if tuple(comp_0 + comp_1) not in token_pair_dict.keys():
                    token_pair_dict[tuple(comp_0 + comp_1)] = 1
                else:
                    token_pair_dict[tuple(comp_0 + comp_1)] += 1

            token_pair_max_ind = max(token_pair_dict, key=lambda inp: token_pair_dict[inp])
            token_pair_max = token_pair_dict[token_pair_max_ind]
            self.token_pair_max_list.append((token_pair_max_ind, token_pair_max))
        if not is_inital_progress:
            self.progress_bar(max(self.token_pair_max_list, key=lambda x: x[1])[1])
        else:
            self.progress_bar(max(self.token_pair_max_list, key=lambda x: x[1])[1], inital=True,
                              initial_max=max(self.token_pair_max_list, key=lambda x: x[1])[1])

    def update_token_list(self, x):
        code_list = []
        for (channel, ind) in zip(x, range(len(x))):
            code = []
            if self.token_pair_max_list[ind][1] > 1:
                max_token_pair = self.token_pair_max_list[ind][0]
                pointer_channel = 0
                while pointer_channel < len(channel):
                    pos = 0
                    point_pos = 0
                    while pos < len(max_token_pair):
                        if pointer_channel + point_pos < len(channel):
                            token = channel[pointer_channel + point_pos]
                            if isinstance(token, tuple):
                                if pos + len(token) <= len(max_token_pair):
                                    if token == tuple(max_token_pair[pos:pos + len(token)]):
                                        if pos + len(token) == len(max_token_pair):
                                            code.append(max_token_pair)
                                            pointer_channel += 2
                                            break
                                        else:
                                            pos += len(token)
                                            point_pos += 1
                                            continue
                            elif token == max_token_pair[pos]:
                                if pos == len(max_token_pair) - 1:
                                    code.append(max_token_pair)
                                    pointer_channel += 2
                                    break
                                else:
                                    pos += 1
                                    point_pos += 1
                                    continue
                        code.append(channel[pointer_channel])
                        pointer_channel += 1
                        break
                code_list.append(code)
            else:
                code_list.append(channel)
        return code_list

    def is_continue(self):
        for pair in self.token_pair_max_list:
            if pair[1] > 1:
                return True
        return False

    def create_word_dict_list(self, x):
        for (channel, ind) in zip(x, range(len(x))):
            rank = {}
            word_dict = {}
            for token in channel:
                rank[token] = rank[token] + 1 if token in rank.keys() else 1
                rank_list = sorted(rank.items(), key=lambda r: r[1])
                for (k, c) in zip(rank_list, range(len(rank_list))):
                    word_dict[k[0]] = c
            self.word_dict_list.append(word_dict)

    def encode_token_pair(self, x):
        code_list = []
        for (channel, ind) in zip(x, range(len(x))):
            code = []
            for i in channel:
                code.append(self.word_dict_list[ind][i])
            code_list.append(code)
        return code_list

    def normalize(self, x):
        for ind in range(len(x)):
            x[ind] = torch.unsqueeze(torch.from_numpy(np.array(x[ind]) / (max(x[ind]) - min(x[ind]))),dim=1)
            self.max_min_code_list.append((max(x[ind]), min(x[ind])))
        return x

    def forward(self, x):
        x = self.channel_split(x)
        self.find_max_token_pair(x, is_inital_progress=True)
        while self.is_continue():
            x = self.update_token_list(x)
            self.find_max_token_pair(x)
        self.create_word_dict_list(x)
        x = self.encode_token_pair(x)
        x = self.normalize(x)
        return x


if __name__ == '__main__':
    train_data = Dataset_ETT_hour(root_path='../dataset')
    test_data = Dataset_ETT_hour(root_path='../dataset', flag='test')
    print(train_data.data_x.shape, test_data.data_x.shape)
    td = np.array(train_data.data_x)
    test = Discretization(num_bins=100)
    bpe = BPEEncode_Channel_Splitting()
    x = test(td)
    x = bpe(x)
    print(x[0].shape)
