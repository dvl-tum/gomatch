import pytorch_lightning as pl
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def init_args():
    # Initial args
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = pl.Trainer.add_argparse_args(parser)  # Add Lightening default args

    # Program args
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--odir", "-o", type=str, default="outputs/train")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--overfit", type=int, default=None)

    # Training
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", "-lr", type=float, default=1e-3)
    parser.add_argument("--resume_version", type=str, default="version0")

    # Matcher args
    parser.add_argument(
        "--matcher_class",
        type=str,
        default="OTMatcherCls",
        choices=["BPnPMatcher", "OTMatcher", "OTMatcherCls"],
    )
    parser.add_argument("--share_kp2d_enc", action="store_true")
    parser.add_argument(
        "--att_layers", type=str, nargs="*", default=["self", "cross", "self"]
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="megadepth")
    parser.add_argument("--dataset_conf", type=str, default="configs/datasets.yml")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--p2d_type", type=str, default="sift")
    parser.add_argument(
        "--p3d_type", choices=["coords", "bvs", "visdesc"], default="bvs"
    )
    parser.add_argument("--npts", type=int, nargs="*", default=[100, 1024])
    parser.add_argument("--outlier_rate", type=float, nargs="*", default=[0.5, 0.5])
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--random_topk", action="store_true")
    parser.add_argument("--inls2d_thres", type=float, default=1)

    # For training BPnPNet
    parser.add_argument("--opt_inliers_only", action="store_true")

    # Cls gt reproj thresh
    parser.add_argument("--rpthres", type=float, default=1)

    return parser


def parse_data_tag(args):
    orate = f"{args.outlier_rate[0]}-{args.outlier_rate[1]}"
    npts = f"{args.npts[0]}-{args.npts[1]}"
    inls_thres = f"inls{args.inls2d_thres}"
    topk = f"top{args.topk}"
    if args.random_topk:
        topk += f"rd{topk}"
    data_tag = f"or{orate}{topk}{args.p2d_type}_{args.p3d_type}{npts}{inls_thres}"
    return data_tag


def parse_model_tag(args):
    model_tag = f"{args.matcher_class}"
    if args.share_kp2d_enc:
        model_tag += ".share2d"
    model_tag += "." + "".join([s[0] for s in args.att_layers])
    return model_tag


def parse_loss_tag(args):
    # Losses
    loss_tag = f"rpthres{args.rpthres}" if args.matcher_class == "OTMatcherCls" else ""
    if args.opt_inliers_only:
        loss_tag += f".opt_inls"
    return loss_tag
