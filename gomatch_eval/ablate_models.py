import argparse
import os
import glob
import torch
import numpy as np

from gomatch.data.dataset_loaders import init_data_loader
from gomatch.utils.evaluator import MatcherEvaluator, summarize_metrics
from gomatch.utils.logger import get_logger

logger = get_logger(level="INFO", name="gomatch_eval")


def ablate_max_orate(
    model_ckpts, data_loader, cache_dir, oracle=False, debug=False, overwrite=False
):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if oracle:
        model_ckpts = [None]

    max_orates = [1.0, 0.75, 0.5, 0.25, 0]
    for ckpt_path in model_ckpts:
        logger.info(f"\n\n\n\n\n>>>>>{ckpt_path}")
        if oracle:
            save_path = os.path.join(cache_dir, "OracleMatcher.npy")
        else:
            save_path = os.path.join(
                cache_dir, os.path.basename(ckpt_path).replace("ckpt", "npy")
            )
        if debug:
            save_path = save_path.replace("npy", "debug.npy")

        # Initialize result dict
        cache_results = {}
        if os.path.exists(save_path):
            # Load cached results
            logger.info(f"Load cache: {save_path}")
            cache_results = np.load(save_path, allow_pickle=True).item()

        # Evaluate per outlier rate
        for max_orate in max_orates:
            data_loader.dataset.outlier_rate = [0, max_orate]
            logger.info(f"\n\n>>>>> orate={data_loader.dataset.outlier_rate} ")
            if max_orate in cache_results and not overwrite:
                summarize_metrics(cache_results[max_orate])
                continue
            evaluator = MatcherEvaluator(ckpt_path=ckpt_path, oracle=oracle)
            data_loader.dataset.p3d_type = evaluator.p3d_type
            evaluator.eval_data_loader(data_loader, debug=debug)
            cache_results[max_orate] = evaluator.metrics
            np.save(save_path, cache_results)
        logger.info(f"Cache results to : {save_path}")


def ablate_archs(
    model_ckpts,
    data_loader,
    cache_dir,
    oracle=False,
    debug=False,
    overwrite=False,
):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if oracle:
        model_ckpts = [None]

    for ckpt_path in model_ckpts:
        logger.info(f"\n\n\n>>>>>{ckpt_path}")

        if oracle:
            save_path = os.path.join(cache_dir, "OracleMatcher.npy")
        else:
            base = os.path.basename(ckpt_path).replace(".ckpt", "")
            save_path = os.path.join(cache_dir, base + ".npy")
        if debug:
            save_path = save_path.replace("npy", "debug.npy")
        if os.path.exists(save_path) and not overwrite:
            # Load cached metrics
            logger.info(f"Load cache: {save_path}")
            metrics = np.load(save_path, allow_pickle=True).item()
            summarize_metrics(metrics)
            continue

        # Eval
        evaluator = MatcherEvaluator(
            ckpt_path=ckpt_path,
            oracle=oracle,
        )
        data_loader.dataset.p3d_type = evaluator.p3d_type
        evaluator.eval_data_loader(data_loader, debug=debug)

        # Save results
        np.save(save_path, evaluator.metrics)
        logger.info(f"Cache results to : {save_path}")


def run_ablation(
    model_dir,
    odir,
    model_pattern,
    root_dir=".",
    oracle=False,
    debug=False,
    ablate_orate=False,
    overwrite=False,
):
    if oracle:
        model_ckpts = [None]
    else:
        # Load ckpts
        model_ckpts = glob.glob(os.path.join(root_dir, model_dir, model_pattern))
        if ablate_orate:
            # Models of interest
            model_ckpts = [f for f in model_ckpts if "best" in f or "BPnPNet" in f]
    logger.info(f"Target model ckpts: {len(model_ckpts)}")

    # Load dataset
    config = dict(
        data_root=os.path.join(root_dir, "data"),
        dataset="megadepth",
        dataset_conf=os.path.join(root_dir, "configs/datasets.yml"),
        p2d_type="sift",
        p3d_type="coords",
        topk=1,
        npts=[10, 1024],
        outlier_rate=[0, 1],
    )
    data_loader = init_data_loader(config, split="test")

    # Cache dir
    dataset = data_loader.dataset
    tag = f"covis{dataset.topk}inls{dataset.inls2d_thres}"
    if dataset.normalized_thres:
        tag += "normth"
    ablation_type = "orate_ablation" if ablate_orate else "arch_ablation"
    cache_dir = os.path.join(root_dir, odir, ablation_type, tag)

    # Eval
    ablate_fn = ablate_max_orate if ablate_orate else ablate_archs
    ablate_fn(
        model_ckpts,
        data_loader,
        cache_dir=cache_dir,
        oracle=oracle,
        debug=debug,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_pattern", type=str, default="*")
    parser.add_argument("--odir", type=str, default="outputs/benchmark_cache_release")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ablate_orate", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    run_ablation(
        model_dir=args.model_dir,
        odir=args.odir,
        model_pattern=args.model_pattern,
        root_dir=args.root_dir,
        oracle=args.oracle,
        ablate_orate=args.ablate_orate,
        debug=args.debug,
        overwrite=args.overwrite,
    )
