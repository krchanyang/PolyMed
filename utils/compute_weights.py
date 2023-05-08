import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import torch


def compute_class_weights(train_y):
    weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_y), y=train_y
    )
    class_weights_dict = dict(zip(np.unique(train_y), weights))
    return class_weights_dict


def compute_sample_weights(train_y):
    sample_weights = compute_sample_weight(class_weight="balanced", y=train_y)
    return sample_weights


def compute_class_weights_torch(train_y):
    class_weights = compute_class_weights(train_y)
    class_weight_list = []
    for idx, (k, v) in enumerate(class_weights.items()):
        if idx == k:
            class_weight_list.append(v)
        else:
            raise Exception(
                "The order of labels in class weight is broken, check the weight dictionary"
            )
    class_weight_list = torch.tensor(np.array(class_weight_list)).type(
        torch.FloatTensor
    )

    return class_weight_list
