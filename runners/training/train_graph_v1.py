import torch
from torch import nn
from tools.Config import G_EMB_DIM, G_OUT_DIM, ATT_HEAD, CONCAT_SIZE, EPOCH
from models.Knowledge_models import (
    GATv2,
    Knowledge_search,
    KnowledgeMLP_v1,
)
import tqdm
from collections import defaultdict
import itertools
from utils.metrics import recall_k, precision_k, f1_k, ndcg_k
import json
import os
from utils.compute_weights import compute_class_weights_torch


class GraphV1TrainingRunner:
    def __init__(
        self,
        train_x,
        train_y,
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
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.word_idx_case = (
            word_idx_case  # polymed.data_variable.word_idx_case['diagnosis']
        )
        self.org_kb_data = org_kb_data  # polymed.org_kb_data
        self.word_idx_total = word_idx_total  # polymed.data_variable.word_idx_total
        self.idx_word_total = idx_word_total  # polymed.data_variable.idx_word_total
        self.word_idx_allkb = word_idx_allkb  # polymed.data_variable.word_idx_allkb
        self.graph = graph  # Training_data().graph
        self.class_weights = args.class_weights
        self.k = args.k
        self.save_base_path = os.path.join(args.save_base_path, args.train_data_type)

    def train(self):
        print("Graph MLP v1 Training Start...")
        model_save_path = os.path.join(self.save_base_path, "Graph_v1")
        os.makedirs(model_save_path, exist_ok=True)

        gat_input_feats = G_EMB_DIM
        gat_output_feats = G_OUT_DIM
        num_heads = ATT_HEAD
        concat_size = CONCAT_SIZE

        dc_input = len(self.train_x[0])
        dc_output = len(self.word_idx_case)

        train_x = torch.tensor(self.train_x).type(torch.FloatTensor).to(self.device)
        train_y = torch.tensor(self.train_y).type(torch.LongTensor).to(self.device)
        test_x = torch.tensor(self.test_x).type(torch.FloatTensor).to(self.device)

        criterion = nn.CrossEntropyLoss()

        if self.class_weights:
            class_weight_list = compute_class_weights_torch(self.train_y).to(
                self.device
            )
            criterion = nn.CrossEntropyLoss(weight=class_weight_list)

        kbsearch = Knowledge_search(
            self.org_kb_data, self.word_idx_total, self.idx_word_total
        )
        search_list = kbsearch.knowledge_sym_search(
            self.train_x, self.idx_word_total, self.word_idx_allkb
        )

        test_search_list = kbsearch.knowledge_sym_search(
            self.test_x, self.idx_word_total, self.word_idx_allkb
        )

        graph = self.graph.to(self.device)

        kg_mlp = KnowledgeMLP_v1(
            input_size=dc_input,
            output_size=dc_output,
            kg_size=gat_output_feats,
            concat_size=concat_size,
        )
        # kg_mlp.apply(self.init_weights)
        kg_mlp.to(self.device)
        # kg_mlp = kg_mlp.to(self.device)

        node_embed = nn.Embedding(graph.number_of_nodes(), gat_input_feats)
        inputs = node_embed.weight
        nn.init.xavier_uniform_(inputs)
        inputs = inputs.to(self.device)

        gat_net = GATv2(gat_input_feats, gat_output_feats, num_heads)
        gat_net.to(self.device)

        optimizer = torch.optim.Adam(
            itertools.chain(
                gat_net.parameters(), node_embed.parameters(), kg_mlp.parameters()
            ),
            lr=0.01,
        )

        train_history = defaultdict(list)
        test_history = defaultdict(list)
        best_result = defaultdict(dict)

        prev_test_recall_1 = 1e-4

        for t in tqdm.tqdm(range(EPOCH)):
            # Train
            kg_mlp.train()
            gat_net.train()

            optimizer.zero_grad()

            graph_emb = gat_net(graph, inputs)

            y_pred = kg_mlp(train_x, graph_emb, search_list)

            loss = criterion(y_pred, train_y)

            loss.backward(retain_graph=True)

            optimizer.step()

            train_history["train_loss"].append(loss.item())

            # Test
            kg_mlp.eval()
            gat_net.eval()

            with torch.no_grad():
                test_pred = kg_mlp(test_x, graph_emb, test_search_list)
                test_pred = test_pred.cpu().detach().numpy()

            for k in self.k:
                test_history[f"recall_{k}"].append(recall_k(test_pred, self.test_y, k))
                test_history[f"precision_{k}"].append(
                    precision_k(test_pred, self.test_y, k)
                )
                test_history[f"f1_{k}"].append(f1_k(test_pred, self.test_y, k))
                test_history[f"ndcg_{k}"].append(ndcg_k(test_pred, self.test_y, k))

            test_recall_1 = recall_k(test_pred, self.test_y, 1)

            if prev_test_recall_1 < test_recall_1:
                prev_test_recall_1 = test_recall_1

                best_result["epoch"] = t + 1
                best_result["train_loss"] = train_history["train_loss"][-1]
                for k in self.k:
                    best_result[f"recall_{k}"] = test_history[f"recall_{k}"][-1]
                    best_result[f"precision_{k}"] = test_history[f"precision_{k}"][-1]
                    best_result[f"f1_{k}"] = test_history[f"f1_{k}"][-1]
                    best_result[f"ndcg_{k}"] = test_history[f"ndcg_{k}"][-1]
                torch.save(
                    {
                        "gatv2": gat_net.state_dict(),
                        "kg_mlp": kg_mlp.state_dict(),
                        "emb": inputs,
                        "graph": graph,
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(model_save_path, "knowledge_mlp_v1.pt"),
                )

                with open(
                    os.path.join(model_save_path, "best_results.json"),
                    "w",
                    encoding="utf-8",
                ) as json_file:
                    json.dump(best_result, json_file, indent="\t")

            if t % 100 == 0:
                print("\n", "=" * 5, "Traning check", "=" * 5)
                print("Recall@1: ", test_history["recall_1"][-1])
                print("Recall@3: ", test_history["recall_3"][-1])
                print("Recall@5: ", test_history["recall_5"][-1])
                print("Loss: ", loss.item())
                print("=" * 23)
        print(f"Best result: \n{best_result}")
        print("Graph MLP v1 Training done and Save the best params...")
