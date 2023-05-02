import torch
from torch import nn
from tools.Config import BLOCK_NUM, LEARNING_RATE, MMT, EPOCH 
from models.MLP_models import Linear_resnet
import tqdm
from collections import defaultdict
from utils.metrics import recall_k, precision_k, f1_k, ndcg_k
import json
import os


class MLPResNetTrainingRunner:
    def __init__(
        self, train_x, train_y, test_x, test_y, word_idx_case, args, device
    ):
        self.device = device
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.word_idx_case = word_idx_case
        self.k = args.k
        self.save_base_path = os.path.join(args.save_base_path, args.train_data_type)

    def train(self):
        print("MLP ResNet Training Start...")
        model_save_path = os.path.join(self.save_base_path, "ResNet")
        os.makedirs(model_save_path, exist_ok=True)

        train_x = torch.tensor(self.train_x).type(torch.FloatTensor).to(self.device)
        train_y = torch.tensor(self.train_y).type(torch.LongTensor).to(self.device)
        test_x = torch.tensor(self.test_x).type(torch.FloatTensor).to(self.device)
        test_y = self.test_y

        resnet_mlp = Linear_resnet(
            input_size=len(train_x[0]),
            output_size=len(self.word_idx_case),
            block_num=BLOCK_NUM,
        )
        resnet_mlp = resnet_mlp.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            resnet_mlp.parameters(), lr=LEARNING_RATE, momentum=MMT
        )

        train_history = defaultdict(list)
        test_history = defaultdict(list)
        best_result = defaultdict(dict)

        prev_test_recall_1 = 1e-4

        for t in tqdm.tqdm(range(EPOCH)):
            # Train
            resnet_mlp.train()
            # simple_mlp.zero_grad()
            optimizer.zero_grad()

            y_pred = resnet_mlp(train_x)
            loss = criterion(y_pred, train_y)

            loss.backward()

            optimizer.step()

            train_history["train_loss"].append(loss.item())

            # Test
            resnet_mlp.eval()
            with torch.no_grad():
                test_pred = resnet_mlp(test_x)
                test_pred = test_pred.cpu().detach().numpy()
            for k in self.k:
                test_history[f"recall_{k}"].append(recall_k(test_pred, test_y, k))
                test_history[f"precision_{k}"].append(precision_k(test_pred, test_y, k))
                test_history[f"f1_{k}"].append(f1_k(test_pred, test_y, k))
                test_history[f"ndcg_{k}"].append(ndcg_k(test_pred, test_y, k))

            test_recall_1 = recall_k(test_pred, test_y, 1)

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
                        "model": resnet_mlp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(model_save_path, "resnet_mlp.pt"),
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

        print("MLP ResNet Training done and Save the best params...")
