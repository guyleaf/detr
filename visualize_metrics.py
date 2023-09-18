import argparse
import os
from pathlib import Path

from util.plot_utils import plot_logs, plot_precision_recall

DEFAULT_KEYS = ["loss", "loss_bbox", "loss_ce", "loss_giou", "class_error", "mAP"]


def get_arg_parser():
    parser = argparse.ArgumentParser("Visualize metrics")
    parser.add_argument(
        "experiments",
        type=str,
        nargs="+",
        help="The path of experiments you'd like to visualize",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--keys",
        type=str,
        nargs="+",
        default=DEFAULT_KEYS,
        help="The keys of metrics you'd like to visualize",
    )
    parser.add_argument(
        "--log_ewm_com",
        type=float,
        default=0,
        help="The decay in terms of center of mass of exponential weighted smoothing",
    )
    parser.add_argument(
        "--pr_naming_scheme",
        type=str,
        default="exp_id",
        help="The name scheme of Precision-Recall plot",
    )

    return parser


def main(args):
    experiments = [Path(experiment) for experiment in args.experiments]
    result = plot_logs(experiments, fields=args.keys, ewm_col=args.ewm_com)
    assert result is not None

    figure, _ = result
    figure.savefig(os.path.join(args.output_dir, "figure.jpg"))
    figure.clear()

    for experiment in experiments:
        output_file = os.path.join(args.output_dir, f"{experiment.name}_pr.jpg")
        eval_dir = experiment / "eval"
        files = list(eval_dir.glob("*.pth"))
        figure, _ = plot_precision_recall(files, naming_scheme=args.naming_scheme)
        figure.savefig(output_file)
        figure.clear()

    print(f"Finished visualizing metrics stored in {args.output_dir}.")


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
