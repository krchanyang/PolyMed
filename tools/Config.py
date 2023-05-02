


# ML best hyperparameter
DT = {'ccp_alpha': 0.0, 'class_weight': 'None', 'criterion': 'gini', 'max_depth': 'None', 'max_features': 'None', 'max_leaf_nodes': 'None', 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1,
      'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 5642, 'splitter': 'best'}

RF = {'bootstrap': 'True', 'ccp_alpha': 0.0, 'class_weight': 'None', 'criterion': 'gini', 'max_depth': 'None', 'max_features': 'auto', 'max_leaf_nodes': 'None', 'max_samples': 'None',
      'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': 'False', 'random_state': 8680,
      'verbose': 0, 'warm_start': 'False'}

EXT = {'bootstrap': 'False', 'ccp_alpha': 0.0, 'class_weight': 'None', 'criterion': 'gini', 'max_depth': 'None', 'max_features': 'auto', 'max_leaf_nodes': 'None', 'max_samples': 'None',
       'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': 'False', 'random_state': 2193,
       'verbose': 0, 'warm_start': 'False'}

GB = {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': 'None', 'learning_rate': 0.15, 'loss': 'deviance', 'max_depth': 4, 'max_features': 1.0, 'max_leaf_nodes': 'None',
      'min_impurity_decrease': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 200, 'n_iter_no_change': 'None', 'random_state': 3327,
      'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': 'False'}

XGB = {'objective': 'multi:softprob', 'use_label_encoder': False, 'base_score': 0.5, 'booster': 'gbtree', 'callbacks': None, 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.9,
        'early_stopping_rounds': None, 'enable_categorical': False, 'eval_metric': None, 'gamma': 0, 'gpu_id': -1, 'grow_policy': 'depthwise', 'importance_type': None, 'interaction_constraints': '',
        'learning_rate': 0.05, 'max_bin': 256, 'max_cat_to_onehot': 4, 'max_delta_step': 0, 'max_depth': 9, 'max_leaves': 0, 'min_child_weight': 1, 'monotone_constraints': '()',
        'n_estimators': 120, 'n_jobs': -1, 'num_parallel_tree': 1, 'predictor': 'auto', 'random_state': 4615, 'reg_alpha': 2, 'reg_lambda': 4, 'sampling_method': 'uniform', 'scale_pos_weight': 49.7,
        'subsample': 0.9, 'tree_method': 'auto', 'validate_parameters': 1, 'verbosity': 0}

LGBM = {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 16, 'min_child_weight': 0.001,
        'min_split_gain': 0.1, 'n_estimators': 100, 'n_jobs': -1, 'num_leaves': 70, 'objective': None, 'random_state': 5519, 'reg_alpha': 0.01, 'reg_lambda': 1e-06, 'silent': 'warn', 'subsample': 1.0,
        'subsample_for_bin': 200000, 'subsample_freq': 0, 'feature_fraction': 1.0, 'bagging_freq': 4, 'bagging_fraction': 0.8}

SVM = {'alpha': 0.002, 'average': False, 'class_weight': None, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.3, 'fit_intercept': False, 'l1_ratio': 0.4300000001, 'learning_rate': 'adaptive',
       'loss': 'hinge', 'max_iter': 1000, 'n_iter_no_change': 5, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 4644, 'shuffle': True, 'tol': 0.001, 'validation_fraction': 0.1,
       'verbose': 0, 'warm_start': False}

LR = {'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 1000, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2',
      'random_state': 6968, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}

# MLP Training Param
EPOCH = 3000

LEARNING_RATE = 0.005
MMT = 0.94

# # ResNet Param
BLOCK_NUM = 30

# # Graph MLP
G_EMB_DIM = 64
G_OUT_DIM = 128
ATT_HEAD = 1
CONCAT_SIZE = 64

# # # Graph MLP v2
KB_REFER_NUM = 5
