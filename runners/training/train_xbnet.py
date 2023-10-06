import joblib
import torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from XBNet.training_utils import training,predict
from XBNet.models import XBNETClassifier
from XBNet.run import run_XBNET
from collections import Counter

class XBNetTrainingRunner:
    def __init__(self, train_x, train_y, test_x, test_y, word_idx_case, args, device):
        self.device = device
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.word_idx_case = word_idx_case
        self.class_weights = args.class_weights
        self.k = args.k
        self.save_base_path = os.path.join(args.save_base_path, args.train_data_type)
        self.model_layer = {
            1: {
                'input': len(train_x[0]),
                'output': 512,
                'bias': 1
            },
            2: {
                'input': 512,
                'output': len(Counter(train_y)),
                'bias': 1
            },
            'activation': {
                'function': 'softmax',
                'dim': 1
            }
        }

    def train(self):
        print("XBNet Training Start...")
        model_save_path = os.path.join(self.save_base_path, "XBNet")
        os.makedirs(model_save_path, exist_ok=True)

        train_x = self.train_x
        train_y = self.train_y
        test_x = self.test_x
        test_y = self.test_y

        le = LabelEncoder()
        train_y = np.array(le.fit_transform(train_y))

        model = XBNETClassifier(train_x, train_y, 2, hyperparam=self.model_layer)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        m, acc, lo, val_ac, val_lo = run_XBNET(train_x, test_x, train_y, test_y, model, criterion, optimizer, 512, 300)
        torch.save(
            {
                "xbnet": m.state_dict(),
                "model_instance": m,
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(model_save_path, "xbnet.pt"),
        )
        print("XBNet Training done and Save the best params...")
