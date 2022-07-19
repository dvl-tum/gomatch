import argparse
import os
import glob
import torch
import numpy as np

from gomatch.data.dataset_loaders import init_data_loader
from gomatch.utils.evaluator import MatcherEvaluator, summarize_metrics
from gomatch.utils.logger import get_logger

logger = get_logger(level="INFO", name="gomatch_eval")


def eval_model(
    data_loader,
    cache_dir,
    ckpt_path=None,
    vismatch=False,
    oracle=False,
    covis_k_nums=[1, 3, 5, 7, 10],
    debug=False,
    overwrite=False,
    prefix=None,
):
    if oracle:
        logger.info(f"\n\n\n>>>>>Oracle Matching")
        save_path = os.path.join(cache_dir, f"OracleMatcher.npy")
    elif vismatch:
        logger.info(f"\n\n\n>>>>>Visual Desc Matching")
        save_path = os.path.join(cache_dir, f"VisMatcher.npy")
    elif ckpt_path:
        logger.info(f"\n\n\n>>>>>{ckpt_path}")
        save_path = os.path.join(
            cache_dir, os.path.basename(ckpt_path).replace("ckpt", "npy")
        )
    if debug:
        save_path = save_path.replace("npy", "debug.npy")
    if prefix:
        save_path = save_path.replace("npy", f"{prefix}.npy")

    # Initialize result dict
    cache_results = {}
    if os.path.exists(save_path):
        # Load cached results
        logger.info(f"Load cache: {save_path}")
        cache_results = np.load(save_path, allow_pickle=True).item()

    # Evaluate per covis k
    for covis_k in covis_k_nums:
        data_loader.dataset.topk = covis_k
        logger.info(f">>>>> covis_k={data_loader.dataset.topk} ")
        if covis_k in cache_results and not overwrite:
            summarize_metrics(cache_results[covis_k])
            continue
        evaluator = MatcherEvaluator(
            vismatch=vismatch, ckpt_path=ckpt_path, oracle=oracle
        )
        data_loader.dataset.p3d_type = evaluator.p3d_type
        evaluator.eval_data_loader(data_loader, debug=debug)
        cache_results[covis_k] = evaluator.metrics
        np.save(save_path, cache_results)
        logger.info(f"Cache results to : {save_path}")


def run_benchmark(
    odir,
    root_dir=".",
    dataset_name="megadepth",
    split="test",
    p2d_type="sift",
    ckpt=None,
    vismatch=False,
    oracle=False,
    merge_before_match=False,
    debug=False,
    overwrite=False,
    covis_k_nums=[1, 3, 5, 7, 10],
    npts_max=1024,
    prefix=None,
):
    assert ckpt or vismatch or oracle

    # Load dataset
    config = dict(
        data_root=os.path.join(root_dir, "data"),
        dataset=dataset_name,
        dataset_conf=os.path.join(root_dir, "configs/datasets.yml"),
        p2d_type=p2d_type,
        p3d_type="visdesc" if vismatch else "coords",
        npts=[10, npts_max],
        outlier_rate=[0, 1],
        merge_p3dm=merge_before_match,
    )
    data_loader = init_data_loader(config, split=split)

    # Cache dir
    dataset = data_loader.dataset
    tag = f"{dataset_name}/{split}/{p2d_type}_inls{dataset.inls2d_thres}"
    if dataset.normalized_thres:
        tag += "normth"
    if dataset.npts[1] != 1024:
        tag += f".mnpts{dataset.npts[1]}"

    tag += ".merge_before_match" if dataset.merge_p3dm else ".merge_after_match"
    cache_dir = os.path.join(root_dir, odir, tag)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Eval
    eval_model(
        data_loader,
        cache_dir,
        ckpt_path=ckpt,
        vismatch=vismatch,
        oracle=oracle,
        debug=debug,
        overwrite=overwrite,
        covis_k_nums=covis_k_nums,
        prefix=prefix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "megadepth",
            "7scenes_sift_v2",
            "7scenes_superpoint_v2",
            "cambridge",
            "cambridge_sift",
        ],
        default="megadepth",
    )
    parser.add_argument("--splits", nargs="*", type=str, default=["test"])
    parser.add_argument("--vismatch", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--p2d_type", type=str, choices=["sift", "superpoint"], default="sift"
    )
    parser.add_argument("--covis_k_nums", type=int, nargs="*", default=[10])
    parser.add_argument("--merge_before_match", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--odir", type=str, default="outputs/benchmark_cache_release")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"Splits : {args.splits}")
    for split in args.splits:
        print(f"\n\n Split = {split}")
        run_benchmark(
            odir=args.odir,
            root_dir=args.root_dir,
            dataset_name=args.dataset,
            split=split,
            p2d_type=args.p2d_type,
            ckpt=args.ckpt,
            vismatch=args.vismatch,
            oracle=args.oracle,
            merge_before_match=args.merge_before_match,
            debug=args.debug,
            overwrite=args.overwrite,
            covis_k_nums=args.covis_k_nums,
            prefix=args.prefix,
        )
