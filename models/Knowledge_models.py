from dgl.nn import GATv2Conv
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class GATv2(nn.Module):
    def __init__(self, in_feats, h_feats, num_heads):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(in_feats, h_feats, num_heads)
        self.conv2 = GATv2Conv(h_feats, h_feats, num_heads)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class KnowledgeMLP_v1(nn.Module):
    def __init__(self, input_size, output_size, kg_size, concat_size):
        super().__init__()
        print(
            f"input_size: {input_size} | output_size: {output_size} | kg_size: {kg_size} | concat_size: {concat_size}"
        )

        self.concat_size = concat_size
        self.layer1 = torch.nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, concat_size),
            nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(nn.Linear(concat_size, output_size))
        self.kg_layer = torch.nn.Sequential(nn.Linear(kg_size, concat_size), nn.ReLU())

    def forward(self, x, kg, search_list):
        knowledge_space = []
        for i_list in search_list:
            if i_list:
                for idx, ea in enumerate(i_list):
                    if idx == 0:
                        kg_data = self.kg_layer(kg[ea])
                    else:
                        kg_data = kg_data + self.kg_layer(kg[ea])
                knowledge_space.append(kg_data.view(-1))
            else:
                knowledge_space.append(torch.zeros(self.concat_size).cuda())

        knowledge_space = torch.stack(knowledge_space)

        h = self.layer1(x)

        h = knowledge_space + h

        h = self.layer2(h)

        return h


class KnowledgeMLP_v2(nn.Module):
    def __init__(self, input_size, output_size, kg_size, concat_size):
        super().__init__()
        self.concat_size = concat_size
        self.layer1 = torch.nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, concat_size),
            nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(nn.Linear(concat_size, output_size))
        self.kg_layer = torch.nn.Sequential(
            nn.Linear(kg_size * 5, concat_size), nn.ReLU()
        )

    def forward(self, x, kg, search_list):
        knowledge_space = []
        for i_list in search_list:
            if i_list:
                for idx, ea in enumerate(i_list):
                    if idx == 0:
                        kg_data = kg[ea]
                    else:
                        kg_data = torch.vstack((kg_data, kg[ea]))
                knowledge_space.append(self.kg_layer(kg_data.view(-1)))
            else:
                knowledge_space.append(torch.zeros(self.concat_size).cuda())

        knowledge_space = torch.stack(knowledge_space)

        h = self.layer1(x)
        h = knowledge_space + h

        h = self.layer2(h)

        return h


class Knowledge_search:
    knowlege_onehot = None

    def __init__(self, knowledge, word_idx_dict, idx_word_dict):
        self.word_idx_dict = word_idx_dict
        self.idx_word_dict = idx_word_dict
        self.gen_knowledge_one_hot(knowledge)

    def gen_knowledge_one_hot(self, knowledge):
        temp_one_hot = []
        for idx in range(len(self.idx_word_dict["diagnosis"])):
            temp_bowl = np.zeros(len(self.word_idx_dict["symptoms"]))
            for sym in knowledge[self.idx_word_dict["diagnosis"][idx]]["symptoms"]:
                temp_bowl[self.word_idx_dict["symptoms"][sym]] = 1
            temp_one_hot.append(temp_bowl)

        self.knowlege_onehot = np.array(temp_one_hot)

    def cos_sim_search(self, query, word_dict, kg_idx, k):
        search_list = []
        for q in query[:, : len(word_dict["symptoms"])]:
            cand = np.where(q > 0)[0]
            ex_idx = np.zeros(len(self.word_idx_dict["symptoms"]))
            for sym in cand:
                ex_idx[self.word_idx_dict["symptoms"][word_dict["symptoms"][sym]]] = 1
            temp_search = []
            temp_metrix = np.vstack([self.knowlege_onehot, ex_idx])
            sim = cosine_similarity(temp_metrix, temp_metrix)
            for dis in np.flip(np.argsort(sim[-1][:-1]))[:k]:
                temp_search.append(kg_idx[self.idx_word_dict["diagnosis"][dis]])
            search_list.append(temp_search)

        return search_list

    def knowledge_sym_search(self, x_data, word_dict, kg_idx):
        knowledge_list = []
        for i in x_data[:, : len(word_dict["symptoms"])]:
            cand = np.where(i > 0)[0]
            temp_cand_list = []
            for sym in cand:
                if word_dict["symptoms"][sym] in kg_idx.keys():
                    temp_cand_list.append(kg_idx[word_dict["symptoms"][sym]])
            knowledge_list.append(temp_cand_list)
        return knowledge_list
