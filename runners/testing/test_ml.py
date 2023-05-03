from collections import defaultdict
from glob import glob
import pandas as pd
import os
import joblib
from catboost import CatBoostClassifier


from utils.metrics import recall_k, precision_k, f1_k, ndcg_k
from utils.constants import (
    ML_MODEL_DICT,
    EXTEND_ML_MODEL_SAVE_PATH,
    KB_EXTEND_ML_MODEL_SAVE_PATH,
    EXTEND_TUNED_ML_MODEL_SAVE_PATH,
    KB_EXTEND_TUNED_ML_MODEL_SAVE_PATH,
    NORM_ML_MODEL_SAVE_PATH,
    NORM_TUNED_ML_MODEL_SAVE_PATH,
)


class MLTestingRunner:
    def __init__(self, test_x, test_y, args, device):
        self.test_x = test_x
        self.test_y = test_y
        self.k = args.k
        self.train_data_type = args.train_data_type
        self.test_data_type = args.test_data_type
        self.device = device

    def test_ml_baseline(self):
        results = defaultdict(list)

        csv_save_name = (
            f"ml_baseline_{self.train_data_type}_{self.test_data_type}_result.csv"
        )

        if self.train_data_type == "norm":
            ml_model_paths = sorted(glob(NORM_ML_MODEL_SAVE_PATH))
            csv_save_path = os.path.join(
                NORM_ML_MODEL_SAVE_PATH.split("*")[0], csv_save_name
            )
        if self.train_data_type == "extend":
            ml_model_paths = sorted(glob(EXTEND_ML_MODEL_SAVE_PATH))
            csv_save_path = os.path.join(
                EXTEND_ML_MODEL_SAVE_PATH.split("*")[0], csv_save_name
            )

        for path in ml_model_paths:
            name = path.split("/")[-1].split(".pkl")[0]
            model_name = ML_MODEL_DICT[name]
            results["model"].append(model_name)

            if "cat" in name:
                trained_model = CatBoostClassifier()
                trained_model.load_model(path)
            else:
                trained_model = joblib.load(path)

            pred_proba = trained_model.predict_proba(self.test_x)

            for k in self.k:
                results[f"recall_{k}"].append(recall_k(pred_proba, self.test_y, k))
                results[f"precision_{k}"].append(
                    precision_k(pred_proba, self.test_y, k)
                )
                results[f"f1_{k}"].append(f1_k(pred_proba, self.test_y, k))
                results[f"ndcg_{k}"].append(ndcg_k(pred_proba, self.test_y, k))

        print(result_dataframe)
        result_dataframe = pd.DataFrame(results)
        result_dataframe.to_csv(csv_save_path, index=False)

    def test_ml_tuned(self):
        results = defaultdict(list)

        csv_save_name = (
            f"ml_tuned_{self.train_data_type}_{self.test_data_type}_result.csv"
        )

        if self.train_data_type == "norm":
            ml_model_paths = sorted(glob(NORM_TUNED_ML_MODEL_SAVE_PATH))
            csv_save_path = os.path.join(
                NORM_TUNED_ML_MODEL_SAVE_PATH.split("*")[0], csv_save_name
            )
        if self.train_data_type == "extend":
            ml_model_paths = sorted(glob(EXTEND_TUNED_ML_MODEL_SAVE_PATH))
            csv_save_path = os.path.join(
                EXTEND_TUNED_ML_MODEL_SAVE_PATH.split("*")[0], csv_save_name
            )

        for path in ml_model_paths:
            name = path.split("/")[-1].split(".pkl")[0]
            model_name = ML_MODEL_DICT[name]
            results["model"].append(model_name)

            if "cat" in name:
                trained_model = CatBoostClassifier()
                trained_model.load_model(path)
            else:
                trained_model = joblib.load(path)

            pred_proba = trained_model.predict_proba(self.test_x)

            for k in self.k:
                results[f"recall_{k}"].append(recall_k(pred_proba, self.test_y, k))
                results[f"precision_{k}"].append(
                    precision_k(pred_proba, self.test_y, k)
                )
                results[f"f1_{k}"].append(f1_k(pred_proba, self.test_y, k))
                results[f"ndcg_{k}"].append(ndcg_k(pred_proba, self.test_y, k))

        print(result_dataframe)
        result_dataframe = pd.DataFrame(results)
        result_dataframe.to_csv(csv_save_path, index=False)
