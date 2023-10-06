import joblib
import torch
from collections import defaultdict
import pandas as pd
import os
from utils.constants import (
    NORM_XBNET_MODEL_SAVE_PATH,
    EXTEND_XBNET_MODEL_SAVE_PATH,
    KB_EXTEND_XBNET_MODEL_SAVE_PATH,
    XBNET_SAVED_MODEL_NAME,
)
from utils.metrics import recall_k, precision_k, f1_k, ndcg_k


class XBNetTestingRunner:
    def __init__(self, test_x, test_y, word_idx_case, args, device):
        self.test_x = test_x
        self.test_y = test_y
        self.word_idx_case = word_idx_case
        self.k = args.k
        self.train_data_type = args.train_data_type
        self.test_data_type = args.test_data_type
        self.device = device

    def test_xbnet(self):
        print("XBNet Evaluation Start...")

        csv_save_name = f"xbnet_{self.train_data_type}_{self.test_data_type}_result.csv"

        if self.train_data_type == "norm":
            model_saved_path = os.path.join(
                NORM_XBNET_MODEL_SAVE_PATH, XBNET_SAVED_MODEL_NAME
            )
            csv_save_path = os.path.join(NORM_XBNET_MODEL_SAVE_PATH, csv_save_name)
        if self.train_data_type == "extend":
            model_saved_path = os.path.join(
                EXTEND_XBNET_MODEL_SAVE_PATH, XBNET_SAVED_MODEL_NAME
            )
            csv_save_path = os.path.join(EXTEND_XBNET_MODEL_SAVE_PATH, csv_save_name)
        if self.train_data_type == "kb_extend":
            model_saved_path = os.path.join(
                KB_EXTEND_XBNET_MODEL_SAVE_PATH, XBNET_SAVED_MODEL_NAME
            )
            csv_save_path = os.path.join(KB_EXTEND_XBNET_MODEL_SAVE_PATH, csv_save_name)

        params = torch.load(model_saved_path)
        xbnet = params['model_instance']
        xbnet.load_state_dict(params["xbnet"])

        # Test
        xbnet.eval()
        with torch.no_grad():
            test_x = torch.tensor(self.test_x).type(torch.FloatTensor)
            test_pred = xbnet(test_x)
            test_pred = test_pred.cpu().detach().numpy()

        test_result = defaultdict(dict)

        for k in self.k:
            test_result[f"recall_{k}"] = recall_k(test_pred, self.test_y, k)
            test_result[f"precision_{k}"] = precision_k(test_pred, self.test_y, k)
            test_result[f"f1_{k}"] = f1_k(test_pred, self.test_y, k)
            test_result[f"ndcg_{k}"] = ndcg_k(test_pred, self.test_y, k)

        result_dataframe = pd.DataFrame.from_dict([test_result])
        print(result_dataframe)
        result_dataframe.to_csv(csv_save_path, index=False)
        print("XBNet Evaluation Done and Save the Results...")
