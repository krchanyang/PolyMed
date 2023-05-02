from tools.PolyMed import PolyMed
from runners.training.train_ml import MLTrainingRunner
from runners.training.train_mlp import MLPTrainingRunner
from runners.training.train_resnet import MLPResNetTrainingRunner
from runners.training.train_graph_v1 import GraphV1TrainingRunner
from runners.training.train_graph_v2 import GraphV2TrainingRunner
from utils.datasets import PolymedDataset
from utils.fix_seed import seed_everything
import torch
import os
import argparse


def main(args):
    os.makedirs("./experiments", exist_ok=True)
    seed_everything(args.seed)

    polymed = PolyMed(
        args.data_dir,
        args.stat_dir,
        args.train_data_type,
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
    if "graph" in args.model_name.lower():
        train_x, train_y, graph = dataset.load_train_data()
    else:
        train_x, train_y = dataset.load_train_data()
    test_x, test_y = dataset.load_test_data()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

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
            word_idx_kb,
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
    args = parser.parse_args()

    main(args)
