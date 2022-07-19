import os

from .pl.trainer import train
from .parse_args import *

from gomatch.utils.logger import get_logger

logger = get_logger(level="INFO", name="train")


def parse_arguments():
    parser = init_args()
    args = parser.parse_args()
    if args.matcher_class == "BPnPMatcher":
        args.opt_inliers_only = True

    # Defined in pl
    args.gpus = [int(args.gpus)]
    args.odir = os.path.join(
        args.odir, args.dataset, f"{args.train_split}_{args.val_split}"
    )
    print(args)
    return args


def init_exp_name(args):
    data_tag = parse_data_tag(args)
    loss_tag = parse_loss_tag(args)
    model_tag = parse_model_tag(args)

    # Exp name
    exp_name = f"{data_tag}/{args.prefix}.{model_tag}_{loss_tag}"
    exp_name += f"/batch{args.batch}_lr{args.lr}"
    print(f"Experiment: {exp_name}")
    return exp_name


def main():
    args = parse_arguments()
    exp_name = init_exp_name(args)
    train(args, exp_name)


if __name__ == "__main__":
    main()
