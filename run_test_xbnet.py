from tools.PolyMed import PolyMed
from utils.datasets import PolymedDataset
from runners.testing.test_xbnet import XBNetTestingRunner
from utils.fix_seed import seed_everything
import torch
import os
import argparse
from utils.translation import str2bool


def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        test_data_type=args.test_data_type,
        model_name=args.model_name,
        is_tuning=False,
        is_training=False,
    )
    
    

    train_x, train_y = dataset.load_train_data()
    test_x, test_y = dataset.load_test_data()

    testing_runner = XBNetTestingRunner(train_x, train_y, test_x, test_y, word_idx_case, args, device)
    testing_runner.test_xbnet()
    
    


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
        "--model_dir",
        type=str,
        default="./models/structure",
        help="Path to the directory where the trained model will be saved or loaded. Default is ./models.",
    )
    parser.add_argument(
        "--param_dir",
        type=str,
        default="./models/parameters",
        help="Path to the directory where the trained model will be saved or loaded. Default is ./models.",
    )
    parser.add_argument(
        "--k",
        type=str,
        default=[1, 3, 5],
        help="Recall@k and Precision@k list, Default is [1, 3, 5]",
    )

    parser.add_argument(
        "--train_data_type", type=str, help='"norm", "extend", "kb_extend"'
    )
    parser.add_argument(
        "--test_data_type", type=str, help='"single", "multi", "unseen"'
    )
    parser.add_argument("--save_base_path", type=str, default="./experiments")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="ml_tuned, ML_tuned, mlp, MLP, res, Res, graphv1, GraphV1, graphv2, GraphV2",
    )
    parser.add_argument("--class_weights", type=str2bool, default="False")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
    "--augmentation_strategy",
    type=str,
    help="Train data augmentation strategies. Supports None, SMOTE, Balance, and Tomek. The default is None.",
    default=None)

    args = parser.parse_args()

    main(args)
