import os
import json
import joblib
import optuna
import cupy

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

from utils.compute_weights import compute_class_weights, compute_sample_weights
from utils.metrics import recall_k


class Model_tuning():
    def __init__(self, train_x, train_y, test_x, test_y, k, train_data_type, device, seed):
        self.device = device
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.k = k
        self.train_data_type = train_data_type
        self.seed = seed
        self.save_base_path = os.path.join("experiments", self.train_data_type)
    
    def tune_extratrees(self):
        model_save_path = os.path.join(self.save_base_path, "ML/tuned")
        os.makedirs(model_save_path, exist_ok=True)
        
        class_weight = compute_class_weights(self.train_y)
        sample_weight = compute_sample_weights(self.train_y)
        
        def etc_objective(trial):
            # max_depth = trial.suggest_int('max_depth', 3, 9)
            params = {
                "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 10),
                "max_features": trial.suggest_categorical('max_features', ["sqrt", "log2"]),
                "min_samples_split": trial.suggest_int('min_samples_split', 2, 12), #
                "min_impurity_decrease": trial.suggest_float('min_impurity_decrease',0.0, 0.01),
                "n_estimators": trial.suggest_int('n_estimators', 50, 500),
                "criterion": trial.suggest_categorical('criterion', ['entropy', 'log_loss']),
                "class_weight": "balanced",
                "random_state": self.seed,
                "n_jobs": -1
            }
            
            # min_weight_fraction_leaf=trial.suggest_uniform('min_weight_fraction_leaf', 0.0,0.4)
            # max_features=trial.suggest_categorical('max_features', ["sqrt", "log2"])
            # max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 15, 25)
            # min_impurity_decrease= trial.suggest_float('min_impurity_decrease',0.0, 0.01)
            # n_estimators = trial.suggest_int('n_estimators', 50, 400)
            # criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            etc = ExtraTreesClassifier(**params)
            # n_estimators = trial.suggest_int('n_estimators', 50, 500)
            # # max_depth = trial.suggest_int('max_depth', 2, 25)
            # min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            # min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            # etc = ExtraTreesClassifier(n_estimators=n_estimators,
            #                 #    max_depth=max_depth,
            #                    min_samples_split=min_samples_split,
            #                    min_samples_leaf=min_samples_leaf,
            #                    random_state=self.seed,
            #                    n_jobs=-1)
            etc.fit(self.train_x, self.train_y)
            y_pred_proba = etc.predict_proba(self.test_x)
            recall_1 = recall_k(y_pred_proba, self.test_y, 1)
            return recall_1
        
        """ExtraTrees"""
        etc_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        etc_study.optimize(etc_objective, n_trials=100)
        etc_best_score = etc_study.best_trial.values
        print(f"ExtraTree Result: {etc_best_score}")
        etc_best_params = etc_study.best_params
        with open(os.path.join(model_save_path, "etc_best_params.json"), 'w', encoding='utf-8') as json_file:
            json.dump(etc_best_params, json_file, indent="\t")  
        tuned_etc = ExtraTreesClassifier(**etc_best_params, random_state=self.seed, n_jobs=-1, class_weight='balanced')
        tuned_etc.fit(self.train_x, self.train_y)
        joblib.dump(tuned_etc, os.path.join(model_save_path, "etc_tuned.pkl"))
        
    def tune_xgboost(self):
        model_save_path = os.path.join(self.save_base_path, "ML/tuned")
        os.makedirs(model_save_path, exist_ok=True)
        
        class_weight = compute_class_weights(self.train_y)
        sample_weight = compute_sample_weights(self.train_y)
        
        def xgboost_objective(trial):
            params = {
                "objective": "multi:softprob",
                "eval_metric":'mlogloss',
                "booster": 'gbtree',
                'tree_method':'gpu_hist',
                'predictor':'gpu_predictor',
                'gpu_id': 0,
                'n_jobs': -1,
                "verbosity": 0,
                "random_state": self.seed,
                "learning_rate": trial.suggest_uniform('learning_rate', 0.0001, 0.99),
                'n_estimators': trial.suggest_int("n_estimators", 100, 10000, step=100),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
                "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-2, 1),
                "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-2, 1),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 1.0, 0.05),     
                'min_child_weight': trial.suggest_int('min_child_weight', 2, 15),
                "gamma": trial.suggest_float("gamma", 0.1, 1.0, log=True),
                # 'num_parallel_tree': trial.suggest_int("num_parallel_tree", 1, 500) 추가하면 느려짐.
            }

            xgboost = XGBClassifier(**params)
            x_train = cupy.array(self.train_x)
            y_train = cupy.array(self.train_y)
            x_test = cupy.array(self.test_x)
            xgboost.fit(x_train, y_train)
            y_pred_proba = xgboost.predict_proba(x_test)

            recall_1 = recall_k(y_pred_proba, self.test_y, 1)
            
            del x_train
            del y_train
            del x_test
            
            return recall_1
        
        """XGBoost"""
        xgboost_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        xgboost_study.optimize(xgboost_objective, n_trials=100)
        xgboost_best_params = xgboost_study.best_params
        with open(os.path.join(model_save_path, "xgboost_best_params.json"), 'w', encoding='utf-8') as json_file:
            json.dump(xgboost_best_params, json_file, indent="\t")
        tuned_xgboost = XGBClassifier(**xgboost_best_params)
        tuned_xgboost.fit(self.train_x, self.train_y, sample_weight=sample_weight)
        joblib.dump(tuned_xgboost, os.path.join(model_save_path,'xgboost_tuned.pkl'))
        
    def tune_catboost(self):
        model_save_path = os.path.join(self.save_base_path, "ML/tuned")
        os.makedirs(model_save_path, exist_ok=True)
        
        class_weights = compute_class_weights(self.train_y)
        sample_weights = compute_sample_weights(self.train_y)
        
        def catboost_objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 100),
                # 'random_strength': trial.suggest_float('random_strength', 1.0, 2.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
                "depth": trial.suggest_int("depth", 1, 12),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli"]
                    
                ),
                "task_type": "GPU",
                "random_seed": self.seed,
                "class_weights": class_weights,
                "verbose": False,
                "thread_count": -1
            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            catboost = CatBoostClassifier(**params)
            catboost.fit(self.train_x, self.train_y, verbose=False)
            y_pred_proba = catboost.predict_proba(self.test_x)
            top1 = recall_k(y_pred_proba, self.test_y, 1)
            return top1
        
        """CatBoost"""
        catboost_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        catboost_study.optimize(catboost_objective, n_trials=100)
        catboost_best_params = catboost_study.best_params
        with open(os.path.join(model_save_path, "catboost_best_params.json"), 'w', encoding='utf-8') as json_file:
            json.dump(catboost_best_params, json_file, indent="\t")
        tuned_catboost = CatBoostClassifier(**catboost_best_params, task_type="GPU", random_seed=self.seed, verbose=False, class_weights=class_weight)    
        tuned_catboost.fit(self.train_x, self.train_y)
        tuned_catboost.save_model(os.path.join(model_save_path,'catboost_tuned.pkl'))