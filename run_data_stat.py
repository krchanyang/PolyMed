from tools.PolyMed import PolyMed
import os
import argparse

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
    default=True,
    help="Display the data statistics on the screen. Default is False.",
)
parser.add_argument(
    "--plotting",
    type=bool,
    default=True,
    help="Save the plots of data statistics as image files. Default is False.",
)
parser.add_argument(
    "--integrity",
    type=bool,
    default=True,
    help="Run the data integrity test and display result. Default is False.",
)

args = parser.parse_args()


def main():
    os.makedirs("./data_stat", exist_ok=True)
    _ = PolyMed(
        args.data_dir,
        args.stat_dir,
        args.data_type,
        args.display,
        args.plotting,
        args.integrity,
    )


if __name__ == "__main__":
    main()
