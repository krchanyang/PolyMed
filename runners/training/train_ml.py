from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
import joblib
import os
from utils.compute_weights import compute_class_weights, compute_sample_weights


class MLTrainingRunner:
    def __init__(self, train_x, train_y, args, device):
        self.device = device
        self.train_x = train_x
        self.train_y = train_y
        self.k = args.k
        self.seed = args.seed
        self.class_weights = args.class_weights
        self.save_base_path = os.path.join(args.save_base_path, args.train_data_type)

    def train(self):
        class_weights = compute_class_weights(self.train_y)
        sample_weights = compute_sample_weights(self.train_y)
        if self.class_weights:
            model_save_path = os.path.join(self.save_base_path, "ML/baseline/cw")
            os.makedirs(model_save_path, exist_ok=True)
            lr = LogisticRegression(random_state=self.seed, class_weight="balanced")
            lr.fit(self.train_x, self.train_y)
            joblib.dump(lr, os.path.join(model_save_path, "lr_cw.pkl"))

            catboost = CatBoostClassifier(
                task_type="GPU",
                random_seed=self.seed,
                verbose=False,
                class_weights=class_weights,
            )
            catboost.fit(self.train_x, self.train_y)
            catboost.save_model(os.path.join(model_save_path, "catboost_cw.pkl"))

            lda = LinearDiscriminantAnalysis()
            lda.fit(self.train_x, self.train_y)
            joblib.dump(lda, os.path.join(model_save_path, "lda_cw.pkl"))

            xgboost = XGBClassifier(seed=self.seed, use_label_encoder=False)
            xgboost.fit(self.train_x, self.train_y, sample_weight=sample_weights)
            joblib.dump(xgboost, os.path.join(model_save_path, "xgboost_cw.pkl"))

            gbc = GradientBoostingClassifier(random_state=self.seed)
            gbc.fit(self.train_x, self.train_y, sample_weight=sample_weights)
            joblib.dump(gbc, os.path.join(model_save_path, "gb_cw.pkl"))

            rf = RandomForestClassifier(
                random_state=self.seed, n_jobs=-1, class_weight="balanced"
            )
            rf.fit(self.train_x, self.train_y)
            joblib.dump(rf, os.path.join(model_save_path, "rf_cw.pkl"))

            et = ExtraTreesClassifier(
                random_state=self.seed, n_jobs=-1, class_weight="balanced"
            )
            et.fit(self.train_x, self.train_y)
            joblib.dump(et, os.path.join(model_save_path, "et_cw.pkl"))

            dt = DecisionTreeClassifier(random_state=self.seed, class_weight="balanced")
            dt.fit(self.train_x, self.train_y)
            joblib.dump(dt, os.path.join(model_save_path, "dt_cw.pkl"))

            knn = KNeighborsClassifier(n_jobs=-1)
            knn.fit(self.train_x, self.train_y)
            joblib.dump(knn, os.path.join(model_save_path, "knn_cw.pkl"))

            nb = GaussianNB()
            nb.fit(self.train_x, self.train_y, sample_weight=sample_weights)
            joblib.dump(nb, os.path.join(model_save_path, "nb_cw.pkl"))

            ada = AdaBoostClassifier(random_state=self.seed)
            ada.fit(self.train_x, self.train_y, sample_weight=sample_weights)
            joblib.dump(ada, os.path.join(model_save_path, "adaboost_cw.pkl"))

            lgbm = LGBMClassifier(
                random_state=self.seed, n_jobs=-1, class_weight="balanced"
            )
            lgbm.fit(self.train_x, self.train_y)
            joblib.dump(lgbm, os.path.join(model_save_path, "lgbm_cw.pkl"))
        else:
            model_save_path = os.path.join(self.save_base_path, "ML/baseline/ncw")
            os.makedirs(model_save_path, exist_ok=True)
            lr = LogisticRegression(random_state=self.seed)
            lr.fit(self.train_x, self.train_y)
            joblib.dump(lr, os.path.join(model_save_path, "lr.pkl"))

            catboost = CatBoostClassifier(
                task_type="GPU",
                random_seed=self.seed,
                verbose=False,
            )
            catboost.fit(self.train_x, self.train_y)
            catboost.save_model(os.path.join(model_save_path, "catboost.pkl"))

            lda = LinearDiscriminantAnalysis()
            lda.fit(self.train_x, self.train_y)
            joblib.dump(lda, os.path.join(model_save_path, "lda.pkl"))

            xgboost = XGBClassifier(seed=self.seed, use_label_encoder=False)
            xgboost.fit(self.train_x, self.train_y)
            joblib.dump(xgboost, os.path.join(model_save_path, "xgboost.pkl"))

            gbc = GradientBoostingClassifier(random_state=self.seed)
            gbc.fit(self.train_x, self.train_y)
            joblib.dump(gbc, os.path.join(model_save_path, "gb.pkl"))

            rf = RandomForestClassifier(random_state=self.seed, n_jobs=-1)
            rf.fit(self.train_x, self.train_y)
            joblib.dump(rf, os.path.join(model_save_path, "rf.pkl"))

            et = ExtraTreesClassifier(random_state=self.seed, n_jobs=-1)
            et.fit(self.train_x, self.train_y)
            joblib.dump(et, os.path.join(model_save_path, "et.pkl"))

            dt = DecisionTreeClassifier(random_state=self.seed)
            dt.fit(self.train_x, self.train_y)
            joblib.dump(dt, os.path.join(model_save_path, "dt.pkl"))

            knn = KNeighborsClassifier(n_jobs=-1)
            knn.fit(self.train_x, self.train_y)
            joblib.dump(knn, os.path.join(model_save_path, "knn.pkl"))

            nb = GaussianNB()
            nb.fit(self.train_x, self.train_y)
            joblib.dump(nb, os.path.join(model_save_path, "nb.pkl"))

            ada = AdaBoostClassifier(random_state=self.seed)
            ada.fit(self.train_x, self.train_y)
            joblib.dump(ada, os.path.join(model_save_path, "adaboost.pkl"))

            lgbm = LGBMClassifier(random_state=self.seed, n_jobs=-1)
            lgbm.fit(self.train_x, self.train_y)
            joblib.dump(lgbm, os.path.join(model_save_path, "lgbm.pkl"))
