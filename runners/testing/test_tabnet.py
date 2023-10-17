import joblib
import torch
from collections import defaultdict
import pandas as pd
import os
from utils.constants import (
    NORM_TABNET_MODEL_SAVE_PATH,
    EXTEND_TABNET_MODEL_SAVE_PATH,
    KB_EXTEND_TABNET_MODEL_SAVE_PATH,
    TABNET_SAVED_MODEL_NAME,
)
from utils.metrics import recall_k, precision_k, f1_k, ndcg_k


class TabNetTestingRunner:
    def __init__(self, test_x, test_y, word_idx_case, args, device):
        self.test_x = test_x
        self.test_y = test_y
        self.word_idx_case = word_idx_case
        self.k = args.k
        self.train_data_type = args.train_data_type
        self.test_data_type = args.test_data_type
        self.augmentation_strategy = args.augmentation_strategy
        self.device = device

    def test_tabnet(self):
        print("TabNet Evaluation Start...")

        csv_save_name = f"tabnet_{self.train_data_type}_{self.augmentation_strategy}_{self.test_data_type}_result.csv"

        if self.train_data_type == "norm":
            tabnet = joblib.load(
                os.path.join(
                    NORM_TABNET_MODEL_SAVE_PATH.format(self.augmentation_strategy),
                    TABNET_SAVED_MODEL_NAME,
                )
            )
            csv_save_path = os.path.join(
                NORM_TABNET_MODEL_SAVE_PATH.format(self.augmentation_strategy),
                csv_save_name,
            )
        if self.train_data_type == "extend":
            tabnet = joblib.load(
                os.path.join(
                    EXTEND_TABNET_MODEL_SAVE_PATH.format(self.augmentation_strategy),
                    TABNET_SAVED_MODEL_NAME,
                )
            )
            csv_save_path = os.path.join(
                EXTEND_TABNET_MODEL_SAVE_PATH.format(self.augmentation_strategy),
                csv_save_name,
            )
        if self.train_data_type == "kb_extend":
            tabnet = joblib.load(
                os.path.join(
                    KB_EXTEND_TABNET_MODEL_SAVE_PATH.format(self.augmentation_strategy),
                    TABNET_SAVED_MODEL_NAME,
                )
            )
            csv_save_path = os.path.join(
                KB_EXTEND_TABNET_MODEL_SAVE_PATH.format(self.augmentation_strategy),
                csv_save_name,
            )

        # Test
        test_pred = tabnet.predict_proba(self.test_x)

        test_result = defaultdict(dict)

        for k in self.k:
            test_result[f"recall_{k}"] = recall_k(test_pred, self.test_y, k)
            test_result[f"precision_{k}"] = precision_k(test_pred, self.test_y, k)
            test_result[f"f1_{k}"] = f1_k(test_pred, self.test_y, k)
            test_result[f"ndcg_{k}"] = ndcg_k(test_pred, self.test_y, k)

        result_dataframe = pd.DataFrame.from_dict([test_result])
        print(result_dataframe)
        result_dataframe.to_csv(csv_save_path, index=False)
        print("TabNet Evaluation Done and Save the Results...")
