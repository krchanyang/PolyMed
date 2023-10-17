import joblib
import torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from collections import Counter

class TabNetTrainingRunner:
    def __init__(self, train_x, train_y, test_x, test_y, word_idx_case, args, device):
        self.device = device
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.word_idx_case = word_idx_case
        self.class_weights = args.class_weights
        self.k = args.k
        self.augmentation_strategy = args.augmentation_strategy
        self.save_base_path = os.path.join(args.save_base_path, args.train_data_type)

    def train(self):
        print("TabNet Training Start...")
        model_save_path = os.path.join(self.save_base_path, "TabNet")
        model_save_path = os.path.join(model_save_path, str(self.augmentation_strategy))

        os.makedirs(model_save_path, exist_ok=True)

        train_x = self.train_x
        train_y = self.train_y
        test_x = self.test_x
        test_y = self.test_y

        clf = TabNetClassifier()  # TabNetRegressor()
        clf.fit(
            train_x, train_y,
            max_epochs=1000,
            patience=300,
            eval_set=[(test_x, test_y)]
        )

        joblib.dump(clf, os.path.join(model_save_path, "tabnet.pkl"))

        print("TabNet Training done and Save the best params...")
