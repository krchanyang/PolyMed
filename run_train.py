from tools.PolyMed import PolyMed
from runners.training.train_ml import MLTrainingRunner
from runners.training.train_mlp import MLPTrainingRunner
from runners.training.train_resnet import MLPResNetTrainingRunner
from runners.training.train_graph_v1 import GraphV1TrainingRunner
from runners.training.train_graph_v2 import GraphV2TrainingRunner
from runners.training.train_xbnet import XBNetTrainingRunner
from runners.training.train_tabnet import TabNetTrainingRunner
from utils.datasets import PolymedDataset
from utils.fix_seed import seed_everything
from tools.imbalance import basic_SMOTE, balance_SMOTE, basic_SMOTE_Tomek
import os
import torch
import argparse
from utils.translation import str2bool


def main(args):
    os.makedirs("./experiments", exist_ok=True)
    seed_everything(args.seed)
    device = f"cuda" if torch.cuda.is_available() else "cpu"

    polymed = PolyMed(
        args.data_dir,
        args.stat_dir,
        args.data_type,
        args.display,
        args.plotting,
        args.integrity,
    )

    word_idx_case = polymed.data_variable.word_idx_case["diagnosis"]
    org_kb_data = polymed.org_kb_data
    word_idx_total = polymed.data_variable.word_idx_total
    idx_word_total = polymed.data_variable.idx_word_total
    word_idx_kb = polymed.data_variable.word_idx_kb
    word_idx_allkb = polymed.data_variable.word_idx_allkb

    dataset = PolymedDataset(
        polymed=polymed,
        train_data_type=args.train_data_type,
        test_data_type=None,
        model_name=args.model_name,
    )
    a_s = args.augmentation_strategy
    print(f"Augmentation Strategy: {a_s}")

    if "graph" in args.model_name.lower():
        train_x, train_y, graph = dataset.load_train_data()
    else:
        train_x, train_y = dataset.load_train_data()
    if a_s is not None:
        a_s = a_s.lower()
        if a_s == "smote":
            train_x, train_y = basic_SMOTE(train_x, train_y)
        if a_s == "balance":
            train_x, train_y = balance_SMOTE(train_x, train_y)
        if a_s == "tomek":
            train_x, train_y = basic_SMOTE_Tomek(train_x, train_y)

    test_x, test_y = dataset.load_test_data()

    print(f"train_x shape: {train_x.shape} | train_y.shape: {train_y.shape}")

    if args.model_name.lower() == "ml":
        training_runner = MLTrainingRunner(train_x, train_y, args, device)
    if args.model_name.lower() == "mlp":
        training_runner = MLPTrainingRunner(
            train_x, train_y, test_x, test_y, word_idx_case, args, device
        )
    if args.model_name.lower() == "res":
        training_runner = MLPResNetTrainingRunner(
            train_x, train_y, test_x, test_y, word_idx_case, args, device
        )
    if args.model_name.lower() == "tabnet":
        training_runner = TabNetTrainingRunner(
            train_x, train_y, test_x, test_y, word_idx_case, args, device
        )
    if args.model_name.lower() == "graphv1":
        training_runner = GraphV1TrainingRunner(
            train_x,
            train_y,
            test_x,
            test_y,
            word_idx_case,
            org_kb_data,
            word_idx_total,
            idx_word_total,
            word_idx_allkb,
            graph,
            args,
            device,
        )
    if args.model_name.lower() == "graphv2":
        training_runner = GraphV2TrainingRunner(
            train_x,
            train_y,
            test_x,
            test_y,
            word_idx_case,
            org_kb_data,
            word_idx_total,
            idx_word_total,
            word_idx_kb,
            word_idx_allkb,
            graph,
            args,
            device,
        )
    if args.model_name.lower() == "xbnet":
        training_runner = XBNetTrainingRunner(
            train_x, train_y, test_x, test_y, word_idx_case, args, device
        )
    training_runner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process some data and generate statistics."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to the directory containing the data to be processed. Default is ./data.",
    )
    parser.add_argument(
        "--stat_dir",
        type=str,
        default="./data_stat",
        help="Path to the directory where the generated statistics will be saved. Default is ./data_stat.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="extend",
        help='Type of data to be processed. Can be either "norm" or "extend". Default is "extend".',
    )
    parser.add_argument(
        "--display",
        type=bool,
        default=False,
        help="Display the data statistics on the screen. Default is False.",
    )
    parser.add_argument(
        "--plotting",
        type=bool,
        default=False,
        help="Save the plots of data statistics as image files. Default is False.",
    )
    parser.add_argument(
        "--integrity",
        type=bool,
        default=False,
        help="Run the data integrity test and display result. Default is False.",
    )
    parser.add_argument(
        "--k",
        type=str,
        default=[1, 3, 5],
        help="Recall@k and Precision@k list, Default is [1, 3, 5]",
    )
    parser.add_argument(
        "--save_base_path",
        type=str,
        default="./experiments",
        help='Base path of the model to save. Default is "./experiments"',
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model Name to train. Supports ml, ML, mlp, MLP, res, Res, graphv1, GraphV1, graphv2, and GraphV2.",
    )
    parser.add_argument(
        "--train_data_type",
        type=str,
        help="Train data type. Supports norm, extend, and kb_extend.",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="Specify GPU number. Default is 0."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Set random state. Default is 42."
    )
    parser.add_argument("--class_weights", type=str2bool, default="False")
    parser.add_argument(
        "--augmentation_strategy",
        type=str,
        help="Train data augmentation strategies. Supports None, SMOTE, Balance, and Tomek. The default is None.",
        default=None,
    )

    args = parser.parse_args()

    main(args)
