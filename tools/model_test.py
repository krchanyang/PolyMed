import numpy as np
import dgl
import torch
from torch import nn
from tools.data_utilities import Data_preprocessing, Training_data
from tools import Config
from models.structure.MLP_models import Disease_classifier, Linear_resnet
from models.structure.Knowledge_models import GATv2, Knowledge_search, KnowledgeMLP_v1, KnowledgeMLP_v2
import tqdm
from collections import defaultdict, Counter
import itertools
from glob import glob
import pandas as pd
import os
import joblib
from catboost import CatBoostClassifier

from utils.constants import ML_MODEL_DICT, EXTEND_ML_MODEL_SAVE_PATH, KB_EXTEND_ML_MODEL_SAVE_PATH, EXTEND_TUNED_ML_MODEL_SAVE_PATH, KB_EXTEND_TUNED_ML_MODEL_SAVE_PATH

class Metric():
    @staticmethod
    def recall_k(proba, ground, k):
        recall_result = []

        top_k = np.flip(np.argsort(proba), axis=1)[:, :k]
        for y_h, y in zip(top_k, ground):
            if type(y) == list:
                recall_result.append(len(set(y) & set(y_h)) / len(y))
            else:
                recall_result.append(y in y_h)

        return np.mean(recall_result)

    @staticmethod
    def precision_k(proba, ground, k):
        precision_result = []

        top_k = np.flip(np.argsort(proba), axis=1)[:, :k]
        for y_h, y in zip(top_k, ground):
            if type(y) == list:
                precision_result.append(len(set(y) & set(y_h)) / k)
            else:
                precision_result.append((y in y_h) / k)
        return np.mean(precision_result)

    @staticmethod
    def f1_k(proba, ground, k):
        r_k = Metric.recall_k(proba, ground, k)
        p_k = Metric.precision_k(proba, ground, k)
        if r_k + p_k == 0:
            f1 = 0
        else:
            f1 = (2 * p_k * r_k) / (r_k + p_k)
        return f1

    @staticmethod
    def dcg(rel, i):
        return rel / np.log2(i+1)

    @staticmethod
    def idcg(rel, data_length):
        accum_idcg = 0
        for i in range(1, data_length+1):
            accum_idcg += Metric.dcg(rel, i)
        return accum_idcg

    @staticmethod
    def ndcg_k(proba, ground, k):
        ndcg_result = []
        target_score = 5 # Suppose all relevance are same(Figures do not affect results)

        top_k = np.flip(np.argsort(proba), axis=1)[:, :k]
        for y_h, y in zip(top_k, ground):
            if type(y) == list:
                accum_dcg = 0
                accum_idcg = Metric.idcg(target_score, len(y))
                for ea_y in y:
                    if ea_y in y_h:
                        accum_dcg += Metric.dcg(target_score, np.where(y_h == ea_y)[0][0] + 1)
            else:
                accum_dcg = 0
                accum_idcg = Metric.idcg(target_score, 1)
                if y in y_h:
                    accum_dcg += Metric.dcg(target_score, np.where(y_h == y)[0][0] + 1)

            if accum_dcg == 0 or accum_idcg == 0:
                ndcg_result.append(0)
            else:
                ndcg_result.append(accum_dcg / accum_idcg)

        return np.mean(ndcg_result)


class Model_test():
    device = None

    def __init__(self, polymed, model_dir, param_dir, k, train_data_type, test_data_type):
        if torch.cuda.is_available(): self.device = torch.device('cuda:0')
        else: self.device = torch.device('cpu')

        self.train_data = Training_data(polymed, 'test')
        self.polymed = polymed
        self.model_dir = model_dir
        self.param_dir = param_dir
        self.k = k
        self.train_data_type = train_data_type
        self.test_data_type = test_data_type

    def test_ml_baseline(self):
        results = {"model": [], "recall_1": [], "recall_3": [], "recall_5": [],
                   "precision_1": [], "precision_3": [], "precision_5": [],
                   "f1_1": [], "f1_3": [], "f1_5": [],
                   "ndcg_1": [], "ndcg_3": [], "ndcg_5": []}
        
        if self.train_data_type == "extend":
            ml_model_paths = sorted(glob(EXTEND_ML_MODEL_SAVE_PATH))
            if self.test_data_type == "single":
                test_x = self.train_data.single_test_x
                test_y = self.train_data.single_test_y
                csv_save_name = "ml_baseline_extend_single_result.csv"
            if self.test_data_type == "multi":
                test_x = self.train_data.multi_test_x
                test_y = self.train_data.multi_test_y
                csv_save_name = "ml_baseline_extend_multi_result.csv"
            if self.test_data_type == "unseen":
                test_x = self.train_data.unseen_test_x
                test_y = self.train_data.unseen_test_y
                csv_save_name = "ml_baseline_extend_unseen_result.csv"

            csv_save_path = os.path.join(EXTEND_ML_MODEL_SAVE_PATH.split("*")[0], csv_save_name)
            
        if self.train_data_type == "kb_extend":
            ml_model_paths = sorted(glob(KB_EXTEND_ML_MODEL_SAVE_PATH))
            if self.test_data_type == "single":
                test_x = self.train_data.kb_single_test_x
                test_y = self.train_data.kb_single_test_y
                csv_save_name = "ml_baseline_kb_extend_single_result.csv"
            if self.test_data_type == "multi":
                test_x = self.train_data.kb_multi_test_x
                test_y = self.train_data.kb_multi_test_y
                csv_save_name = "ml_baseline_kb_extend_multi_result.csv"
            if self.test_data_type == "unseen":
                test_x = self.train_data.kb_unseen_test_x
                test_y = self.train_data.kb_unseen_test_y
                csv_save_name = "ml_baseline_kb_extend_unseen_result.csv"
            
            csv_save_path = os.path.join(KB_EXTEND_ML_MODEL_SAVE_PATH.split("*")[0], csv_save_name)
            
        for path in ml_model_paths:
            name = path.split("/")[-1].split(".pkl")[0]
            model_name = ML_MODEL_DICT[name]
            results["model"].append(model_name)
            
            if "cat" in name:
                trained_model = CatBoostClassifier()
                trained_model.load_model(path)
            else: trained_model = joblib.load(path)
            
            pred_proba = trained_model.predict_proba(test_x)
            
            for k in self.k:
                results[f'recall_{k}'].append(Metric.recall_k(pred_proba, test_y, k))
                results[f'precision_{k}'].append(Metric.precision_k(pred_proba, test_y, k))
                results[f'f1_{k}'].append(Metric.f1_k(pred_proba, test_y, k))
                results[f'ndcg_{k}'].append(Metric.ndcg_k(pred_proba, test_y, k))
                
        result_dataframe = pd.DataFrame(results)
        result_dataframe.to_csv(csv_save_path, index=False)
            
    def test_ml_tuned(self):
        results = {"model": [], "recall_1": [], "recall_3": [], "recall_5": [],
                   "precision_1": [], "precision_3": [], "precision_5": [],
                   "f1_1": [], "f1_3": [], "f1_5": [],
                   "ndcg_1": [], "ndcg_3": [], "ndcg_5": []}
        
        if self.train_data_type == "extend":
            ml_model_paths = sorted(glob(EXTEND_TUNED_ML_MODEL_SAVE_PATH))
            if self.test_data_type == "single":
                test_x = self.train_data.single_test_x
                test_y = self.train_data.single_test_y
                csv_save_name = "ml_baseline_extend_single_result.csv"
            if self.test_data_type == "multi":
                test_x = self.train_data.multi_test_x
                test_y = self.train_data.multi_test_y
                csv_save_name = "ml_baseline_extend_multi_result.csv"
            if self.test_data_type == "unseen":
                test_x = self.train_data.unseen_test_x
                test_y = self.train_data.unseen_test_y
                csv_save_name = "ml_baseline_extend_unseen_result.csv"

            csv_save_path = os.path.join(EXTEND_TUNED_ML_MODEL_SAVE_PATH.split("*")[0], csv_save_name)
            
        if self.train_data_type == "kb_extend":
            ml_model_paths = sorted(glob(KB_EXTEND_TUNED_ML_MODEL_SAVE_PATH))
            if self.test_data_type == "single":
                test_x = self.train_data.kb_single_test_x
                test_y = self.train_data.kb_single_test_y
                csv_save_name = "ml_baseline_kb_extend_single_result.csv"
            if self.test_data_type == "multi":
                test_x = self.train_data.kb_multi_test_x
                test_y = self.train_data.kb_multi_test_y
                csv_save_name = "ml_baseline_kb_extend_multi_result.csv"
            if self.test_data_type == "unseen":
                test_x = self.train_data.kb_unseen_test_x
                test_y = self.train_data.kb_unseen_test_y
                csv_save_name = "ml_baseline_kb_extend_unseen_result.csv"
            
            csv_save_path = os.path.join(KB_EXTEND_TUNED_ML_MODEL_SAVE_PATH.split("*")[0], csv_save_name)
            
        for path in ml_model_paths:
            name = path.split("/")[-1].split(".pkl")[0]
            model_name = ML_MODEL_DICT[name]
            results["model"].append(model_name)
            
            if "cat" in name:
                trained_model = CatBoostClassifier()
                trained_model.load_model(path)
            else: trained_model = joblib.load(path)
            
            pred_proba = trained_model.predict_proba(test_x)
            
            for k in self.k:
                results[f'recall_{k}'].append(Metric.recall_k(pred_proba, test_y, k))
                results[f'precision_{k}'].append(Metric.precision_k(pred_proba, test_y, k))
                results[f'f1_{k}'].append(Metric.f1_k(pred_proba, test_y, k))
                results[f'ndcg_{k}'].append(Metric.ndcg_k(pred_proba, test_y, k))
                
        result_dataframe = pd.DataFrame(results)
        result_dataframe.to_csv(csv_save_path, index=False)

    def test_mlp(self):
        print('Simple MLP Evaluation Start...')

        mlp_single_test_x = torch.tensor(self.train_data.single_test_x).type(torch.FloatTensor).to(self.device)
        mlp_unseen_test_x = torch.tensor(self.train_data.unseen_test_x).type(torch.FloatTensor).to(self.device)
        mlp_multi_test_x = torch.tensor(self.train_data.multi_test_x).type(torch.FloatTensor).to(self.device)

        simple_mlp = Disease_classifier(input_size=len(self.train_data.single_test_x[0]), output_size=len(self.polymed.data_variable.word_idx_case['diagnosis']))
        simple_mlp = simple_mlp.to(self.device)
        params = torch.load(self.param_dir + '/simple_mlp.pth')
        simple_mlp.load_state_dict(params)

        test_result = {}

        # Test
        simple_mlp.eval()
        for test_name, test_dataset, ground_truth in zip(['single', 'unseen', 'multi'], [mlp_single_test_x, mlp_unseen_test_x, mlp_multi_test_x], [self.train_data.single_test_y, self.train_data.unseen_test_y, self.train_data.multi_test_y]):
            test_result[test_name] = {}
            test_pred = simple_mlp(test_dataset)
            test_pred = test_pred.cpu().detach().numpy()

            for k in self.k:
                test_result[test_name][f'recall_{k}'] = Metric.recall_k(test_pred, ground_truth, k)
                test_result[test_name][f'precision_{k}'] = Metric.precision_k(test_pred, ground_truth, k)
                test_result[test_name][f'f1_{k}'] = Metric.f1_k(test_pred, ground_truth, k)
                test_result[test_name][f'ndcg_{k}'] = Metric.ndcg_k(test_pred, ground_truth, k)

        print(test_result)
        print('Simple MLP Evaluation Done and Save the Results...')


    def test_mlp_resnet(self):
        print('MLP ResNet Evaluation Start...')

        mlp_single_test_x = torch.tensor(self.train_data.single_test_x).type(torch.FloatTensor).to(self.device)
        mlp_unseen_test_x = torch.tensor(self.train_data.unseen_test_x).type(torch.FloatTensor).to(self.device)
        mlp_multi_test_x = torch.tensor(self.train_data.multi_test_x).type(torch.FloatTensor).to(self.device)

        resnet_mlp = Linear_resnet(input_size=len(self.train_data.single_test_x[0]), output_size=len(self.polymed.data_variable.word_idx_case['diagnosis']), block_num=Config.BLOCK_NUM)
        resnet_mlp = resnet_mlp.to(self.device)
        params = torch.load(self.param_dir + '/resnet_mlp.pth')
        resnet_mlp.load_state_dict(params)

        test_result = {}

        # Test
        resnet_mlp.eval()
        for test_name, test_dataset, ground_truth in zip(['single', 'unseen', 'multi'], [mlp_single_test_x, mlp_unseen_test_x, mlp_multi_test_x],
                                                         [self.train_data.single_test_y, self.train_data.unseen_test_y, self.train_data.multi_test_y]):
            test_result[test_name] = {}
            test_pred = resnet_mlp(test_dataset)
            test_pred = test_pred.cpu().detach().numpy()

            for k in self.k:
                test_result[test_name][f'recall_{k}'] = Metric.recall_k(test_pred, ground_truth, k)
                test_result[test_name][f'precision_{k}'] = Metric.precision_k(test_pred, ground_truth, k)
                test_result[test_name][f'f1_{k}'] = Metric.f1_k(test_pred, ground_truth, k)
                test_result[test_name][f'ndcg_{k}'] = Metric.ndcg_k(test_pred, ground_truth, k)

        print(test_result)
        print('MLP ResNet Test done...')


    def test_graph_mlp_v1(self):
        print('Graph MLP v1 Training Start...')
        gat_input_feats = Config.G_EMB_DIM
        gat_output_feats = Config.G_OUT_DIM
        num_heads = Config.ATT_HEAD
        concat_size = Config.CONCAT_SIZE

        dc_input = len(self.train_data.kb_single_test_x[0])
        dc_output = len(self.polymed.data_variable.word_idx_case['diagnosis'])

        mlp_single_test_x = torch.tensor(self.train_data.kb_single_test_x).type(torch.FloatTensor).to(self.device)
        mlp_unseen_test_x = torch.tensor(self.train_data.kb_unseen_test_x).type(torch.FloatTensor).to(self.device)
        mlp_multi_test_x = torch.tensor(self.train_data.kb_multi_test_x).type(torch.FloatTensor).to(self.device)

        kbsearch = Knowledge_search(self.polymed.org_kb_data, self.polymed.data_variable.word_idx_total, self.polymed.data_variable.idx_word_total)
        single_search_list = kbsearch.knowledge_sym_search(self.train_data.kb_single_test_x, self.polymed.data_variable.idx_word_total, self.polymed.data_variable.word_idx_kb)
        unseen_search_list = kbsearch.knowledge_sym_search(self.train_data.kb_unseen_test_x, self.polymed.data_variable.idx_word_total, self.polymed.data_variable.word_idx_kb)
        multi_search_list = kbsearch.knowledge_sym_search(self.train_data.kb_multi_test_x, self.polymed.data_variable.idx_word_total, self.polymed.data_variable.word_idx_kb)

        graph = self.train_data.graph.to(self.device)

        params = torch.load(self.param_dir + '/knowledge_mlp_v1.pth')

        kg_mlp = KnowledgeMLP_v1(input_size=dc_input, output_size=dc_output, kg_size=gat_output_feats, concat_size=concat_size)
        kg_mlp.load_state_dict(params['kg_mlp'])
        kg_mlp = kg_mlp.to(self.device)

        inputs = params['emb']
        inputs = inputs.to(self.device)

        gat_net = GATv2(gat_input_feats, gat_output_feats, num_heads)
        gat_net.load_state_dict(params['gatv2'])
        gat_net = gat_net.to(self.device)

        test_result = {}

        # Test
        kg_mlp.eval()
        gat_net.eval()

        graph_emb = gat_net(graph, inputs)

        for test_name, test_dataset, ground_truth, search_list in zip(['single', 'unseen', 'multi'],
                                                                      [mlp_single_test_x, mlp_unseen_test_x, mlp_multi_test_x],
                                                                      [self.train_data.kb_single_test_y, self.train_data.kb_unseen_test_y, self.train_data.kb_multi_test_y],
                                                                      [single_search_list, unseen_search_list, multi_search_list]):
            test_result[test_name] = {}
            test_pred = kg_mlp(test_dataset, graph_emb, search_list)

            test_pred = test_pred.cpu().detach().numpy()

            for k in self.k:
                test_result[test_name][f'recall_{k}'] = Metric.recall_k(test_pred, ground_truth, k)
                test_result[test_name][f'precision_{k}'] = Metric.precision_k(test_pred, ground_truth, k)
                test_result[test_name][f'f1_{k}'] = Metric.f1_k(test_pred, ground_truth, k)
                test_result[test_name][f'ndcg_{k}'] = Metric.ndcg_k(test_pred, ground_truth, k)

        print(test_result)
        print('Graph MLP v1 Evaluation Done and Save the Results...')


    def test_graph_mlp_v2(self):
        print('Graph MLP v2 Evaluation start...')
        gat_input_feats = Config.G_EMB_DIM
        gat_output_feats = Config.G_OUT_DIM
        num_heads = Config.ATT_HEAD
        knowledge_k = Config.KB_REFER_NUM
        concat_size = Config.CONCAT_SIZE

        dc_input = len(self.train_data.kb_single_test_x[0])
        dc_output = len(self.polymed.data_variable.word_idx_case['diagnosis'])

        mlp_single_test_x = torch.tensor(self.train_data.kb_single_test_x).type(torch.FloatTensor).to(self.device)
        mlp_unseen_test_x = torch.tensor(self.train_data.kb_unseen_test_x).type(torch.FloatTensor).to(self.device)
        mlp_multi_test_x = torch.tensor(self.train_data.kb_multi_test_x).type(torch.FloatTensor).to(self.device)

        kbsearch = Knowledge_search(self.polymed.org_kb_data, self.polymed.data_variable.word_idx_total, self.polymed.data_variable.idx_word_total)
        single_search_list = kbsearch.cos_sim_search(self.train_data.kb_single_test_x,
                                                     self.polymed.data_variable.idx_word_total,
                                                     self.polymed.data_variable.word_idx_allkb,
                                                     knowledge_k)
        unseen_search_list = kbsearch.cos_sim_search(self.train_data.kb_unseen_test_x,
                                                     self.polymed.data_variable.idx_word_total,
                                                     self.polymed.data_variable.word_idx_allkb,
                                                     knowledge_k)
        multi_search_list = kbsearch.cos_sim_search(self.train_data.kb_multi_test_x,
                                                    self.polymed.data_variable.idx_word_total,
                                                    self.polymed.data_variable.word_idx_allkb,
                                                    knowledge_k)

        params = torch.load(self.param_dir + '/knowledge_mlp_v2.pth')

        graph = self.train_data.graph.to(self.device)

        kg_mlp = KnowledgeMLP_v2(input_size=dc_input, output_size=dc_output, kg_size=gat_output_feats, concat_size=concat_size)
        kg_mlp.load_state_dict(params['kg_mlp'])
        kg_mlp = kg_mlp.to(self.device)

        inputs = params['emb']
        inputs = inputs.to(self.device)

        gat_net = GATv2(gat_input_feats, gat_output_feats, num_heads)
        gat_net.load_state_dict(params['gatv2'])
        gat_net = gat_net.to(self.device)

        test_result = {}

        # Test
        kg_mlp.eval()
        gat_net.eval()

        graph_emb = gat_net(graph, inputs)

        for test_name, test_dataset, ground_truth, search_list in zip(['single', 'unseen', 'multi'],
                                                                      [mlp_single_test_x, mlp_unseen_test_x, mlp_multi_test_x],
                                                                      [self.train_data.kb_single_test_y, self.train_data.kb_unseen_test_y, self.train_data.kb_multi_test_y],
                                                                      [single_search_list, unseen_search_list, multi_search_list]):
            test_result[test_name] = {}
            test_pred = kg_mlp(test_dataset, graph_emb, search_list)
            test_pred = test_pred.cpu().detach().numpy()

            for k in self.k:
                test_result[test_name][f'recall_{k}'] = Metric.recall_k(test_pred, ground_truth, k)
                test_result[test_name][f'precision_{k}'] = Metric.precision_k(test_pred, ground_truth, k)
                test_result[test_name][f'f1_{k}'] = Metric.f1_k(test_pred, ground_truth, k)
                test_result[test_name][f'ndcg_{k}'] = Metric.ndcg_k(test_pred, ground_truth, k)

        print(test_result)
        print('Graph MLP v2 Evaluation Done and Save the Results...')