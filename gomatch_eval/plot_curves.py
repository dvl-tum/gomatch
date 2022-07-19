from argparse import Namespace
from enum import Enum
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from gomatch.utils.metrics import summarize_metrics
from gomatch.utils.logger import get_logger

logger = get_logger(level="INFO", name="plot")


class Color(Enum):
    GREEN = "#1a6e35"
    BRIGHTGREEN = "#7bfc03"
    LIGHTGRASS = "#c9eb34"
    BROWN = "#a16438"
    YELLOW = "#edbb24"
    PURPLE = "#b157fa"
    LIGHTPURPLE = "#c2cff2"
    GREY = "#9da3a3"
    BLUE = "#4278f5"
    SKY = "#94faff"
    ORANGE = "#f58245"
    CYAN = "#08bdab"
    ROSE = "#eb659d"
    PINK = "#ffd4d4"
    RED = "#db0f0f"
    BLACK = "#000000"


def load_ablation_cache(cache_path, auc_thresholds, key_name="Covis k"):
    metrics = np.load(cache_path, allow_pickle=True).item()
    ablat_keys = list(metrics.keys())
    auc_thres = []
    aucs = []
    for key in ablat_keys:
        logger.info(f">>{key_name}:{key}")
        reproj_auc = summarize_metrics(
            metrics[key],
            auc_thresholds=auc_thresholds,
        )
        aucs.append(reproj_auc)
    aucs = np.array(aucs)
    return ablat_keys, aucs


def plot_orate_ablation_curves(model_results, model_colors, max_orates, auc_thresholds):
    plt.figure(figsize=(18, 5))
    xrange = max_orates
    num_col = len(model_results)
    for idx, th in enumerate(auc_thresholds):
        plt.subplot(1, len(auc_thresholds), idx + 1)
        ymax = 0
        for name, color in model_colors.items():
            aucs = model_results[name][:, idx]
            ymax = max(ymax, aucs.max())
            plt.plot(xrange, aucs, color=color.value, ls="-", linewidth=6, label=name)
        plt.xticks(xrange)
        ylimit = 10 * (ymax // 10 + 1)
        plt.yticks(np.arange(0, ylimit, ylimit // 10))
        #         plt.ylim(0, 10 * (ymax // 10 + 1))
        plt.xlabel("Max Outlier Rate", fontsize=24)
        plt.ylabel(f"AUC @{th}px (%)", fontsize=26)
        plt.grid()
        plt.tick_params(axis="both", which="major", labelsize=24)
    #     plt.legend(fontsize=26, ncol=4, loc='lower center', bbox_to_anchor=(- 0.9, -0.42, 0, 0))
    plt.legend(fontsize=24, bbox_to_anchor=(1.8, 0.73, 0, 0))
    plt.subplots_adjust(wspace=0.35)
