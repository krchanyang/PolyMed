import torch
from collections import defaultdict
import pandas as pd
from models.Knowledge_models import GATv2, Knowledge_search, KnowledgeMLP_v1
import os
from tools.Config import G_EMB_DIM, G_OUT_DIM, ATT_HEAD, CONCAT_SIZE
from utils.constants import (
    EXTEND_GRAPH_V1_MODEL_SAVE_PATH,
    KB_EXTEND_GRAPH_V1_MODEL_SAVE_PATH,
    GRAPH_V1_SAVED_MODEL_NAME,
)
from utils.metrics import recall_k, precision_k, f1_k, ndcg_k


class GraphV1TestingRunner:
    def __init__(
        self,
        test_x,
        test_y,
        word_idx_case,
        org_kb_data,
        word_idx_total,
        idx_word_total,
        word_idx_allkb,
        graph,
        args,
        device,
    ):
        self.device = device
        self.test_x = test_x
        self.test_y = test_y

        self.word_idx_case = (
            word_idx_case  # polymed.data_variable.word_idx_case['diagnosis']
        )
        self.org_kb_data = org_kb_data  # polymed.org_kb_data
        self.word_idx_total = word_idx_total  # polymed.data_variable.word_idx_total
        self.idx_word_total = idx_word_total  # polymed.data_variable.idx_word_total
        self.word_idx_allkb = word_idx_allkb  # polymed.data_variable.word_idx_kb
        self.graph = graph  # Training_data().graph
        self.train_data_type = args.train_data_type
        self.test_data_type = args.test_data_type
        self.k = args.k
        self.augmentation_strategy = args.augmentation_strategy

    def test_graph_mlp_v1(self):
        print("Graph MLP v1 Evaluation Start...")
        gat_input_feats = G_EMB_DIM
        gat_output_feats = G_OUT_DIM
        num_heads = ATT_HEAD
        concat_size = CONCAT_SIZE

        dc_input = len(self.test_x[0])
        dc_output = len(self.word_idx_case)

        csv_save_name = f"graph_v1_{self.train_data_type}_{self.augmentation_strategy}_{self.test_data_type}_result.csv"

        if self.train_data_type == "extend":
            model_saved_path = os.path.join(
                EXTEND_GRAPH_V1_MODEL_SAVE_PATH.format(self.augmentation_strategy),
                GRAPH_V1_SAVED_MODEL_NAME,
            )
            csv_save_path = os.path.join(
                EXTEND_GRAPH_V1_MODEL_SAVE_PATH.format(self.augmentation_strategy),
                csv_save_name,
            )
        if self.train_data_type == "kb_extend":
            model_saved_path = os.path.join(
                KB_EXTEND_GRAPH_V1_MODEL_SAVE_PATH.format(self.augmentation_strategy),
                GRAPH_V1_SAVED_MODEL_NAME,
            )
            csv_save_path = os.path.join(
                KB_EXTEND_GRAPH_V1_MODEL_SAVE_PATH.format(self.augmentation_strategy),
                csv_save_name,
            )

        test_x = torch.tensor(self.test_x).type(torch.FloatTensor).to(self.device)

        kbsearch = Knowledge_search(
            self.org_kb_data, self.word_idx_total, self.idx_word_total
        )
        search_list = kbsearch.knowledge_sym_search(
            self.test_x, self.idx_word_total, self.word_idx_allkb
        )

        params = torch.load(model_saved_path, map_location=self.device)
        graph = params["graph"]

        kg_mlp = KnowledgeMLP_v1(
            input_size=dc_input,
            output_size=dc_output,
            kg_size=gat_output_feats,
            concat_size=concat_size,
        )
        kg_mlp.load_state_dict(params["kg_mlp"])
        kg_mlp = kg_mlp.to(self.device)

        inputs = params["emb"]
        inputs = inputs.to(self.device)

        gat_net = GATv2(gat_input_feats, gat_output_feats, num_heads)
        gat_net.load_state_dict(params["gatv2"])
        gat_net = gat_net.to(self.device)

        test_result = defaultdict(list)

        # Test
        kg_mlp.eval()
        gat_net.eval()

        with torch.no_grad():
            graph_emb = gat_net(graph, inputs)
            test_pred = kg_mlp(test_x, graph_emb, search_list)
            test_pred = test_pred.cpu().detach().numpy()

        for k in self.k:
            test_result[f"recall_{k}"] = recall_k(test_pred, self.test_y, k)
            test_result[f"precision_{k}"] = precision_k(test_pred, self.test_y, k)
            test_result[f"f1_{k}"] = f1_k(test_pred, self.test_y, k)
            test_result[f"ndcg_{k}"] = ndcg_k(test_pred, self.test_y, k)
        result_dataframe = pd.DataFrame.from_dict([test_result])
        print(result_dataframe)
        result_dataframe.to_csv(csv_save_path, index=False)
        print("Graph MLP v1 Evaluation Done and Save the Results...")
