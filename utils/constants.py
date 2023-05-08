ML_MODEL_DICT = {
    "lr": "LogisticRegression",
    "catboost": "CatBoostClassifier",
    "lda": "LinearDiscriminantAnalysis",
    "xgboost": "XGBClassifier",
    "gb": "GradientBoostingClassifier",
    "rf": "RandomForestClassifier",
    "et": "ExtraTreesClassifier",
    "dt": "DecisionTreeClassifier",
    "knn": "KNeighborsClassifier",
    "nb": "GaussianNB",
    "adaboost": "AdaBoostClassifier",
    "lgbm": "LGBMClassifier",
}

NORM_ML_MODEL_SAVE_PATH = "./experiments/norm/ML/baseline/*.pkl"
EXTEND_ML_MODEL_SAVE_PATH = "./experiments/extend/ML/baseline/*.pkl"
KB_EXTEND_ML_MODEL_SAVE_PATH = "./experiments/kb_extend/ML/baseline/ncw/*.pkl"
KB_EXTEND_CLASS_WEIGHTS_ML_MODEL_SAVE_PATH = (
    "./experiments/kb_extend/ML/baseline/cw/*.pkl"
)

EXTEND_TUNED_ML_MODEL_SAVE_PATH = "./experiments/extend/ML/tuned/*.pkl"
KB_EXTEND_TUNED_ML_MODEL_SAVE_PATH = "./experiments/kb_extend/ML/tuned/ncw/*.pkl"
KB_EXTEND_TUNED_CLASS_WEIGHTS_ML_MODEL_SAVE_PATH = "./experiments/kb_extend/ML/tuned/cw/*.pkl"
NORM_TUNED_ML_MODEL_SAVE_PATH = "./experiments/norm/ML/tuned/*.pkl"


NORM_MLP_MODEL_SAVE_PATH = "./experiments/norm/MLP"
EXTEND_MLP_MODEL_SAVE_PATH = "./experiments/extend/MLP"
KB_EXTEND_MLP_MODEL_SAVE_PATH = "./experiments/kb_extend/MLP"
MLP_SAVED_MODEL_NAME = "simple_mlp.pt"

NORM_RESNET_MODEL_SAVE_PATH = "./experiments/norm/ResNet"
EXTEND_RESNET_MODEL_SAVE_PATH = "./experiments/extend/ResNet"
KB_EXTEND_RESNET_MODEL_SAVE_PATH = "./experiments/kb_extend/ResNet"
RESNET_SAVED_MODEL_NAME = "resnet_mlp.pt"

EXTEND_GRAPH_V1_MODEL_SAVE_PATH = "./experiments/extend/Graph_v1"
KB_EXTEND_GRAPH_V1_MODEL_SAVE_PATH = "./experiments/kb_extend/Graph_v1"
GRAPH_V1_SAVED_MODEL_NAME = "knowledge_mlp_v1.pt"

EXTEND_GRAPH_V2_MODEL_SAVE_PATH = "./experiments/extend/Graph_v2"
KB_EXTEND_GRAPH_V2_MODEL_SAVE_PATH = "./experiments/kb_extend/Graph_v2"
GRAPH_V2_SAVED_MODEL_NAME = "knowledge_mlp_v2.pt"
