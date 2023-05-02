import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


def compute_class_weights(train_y):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_y),
        y=train_y
    )
    
    class_weights_dict = dict(zip(np.unique(train_y), weights))
    
    return class_weights_dict

def compute_sample_weights(train_y):
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=train_y
    )
    return sample_weights