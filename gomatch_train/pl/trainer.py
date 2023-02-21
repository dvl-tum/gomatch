from collections import defaultdict
import os
import torch
import pytorch_lightning as pl

from .logger import TensorBoardLoggerMetrics
from gomatch import models
from gomatch.data.dataset_loaders import init_data_loader
from gomatch.utils.loss import compute_loss
from gomatch.utils.metrics import compute_metrics_batch


class MatcherTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        if config.matcher_class == "BPnPMatcher":
            self.matcher = models.BPnPMatcher()
        else:
            self.matcher = vars(models)[config.matcher_class](
                p3d_type=config.p3d_type,
                share_kp2d_enc=config.share_kp2d_enc,
                att_layers=config.att_layers,
            )
        self.cls = isinstance(self.matcher, models.OTMatcherCls)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def parse_data(self, data):
        inputs = [data["pts2dm"], data["idx2d"], data["pts3dm"], data["idx3d"]]
        return inputs

    def compute_loss_and_metrics(self, data):
        inputs = self.parse_data(data)
        preds = self.matcher(*inputs)
        losses = compute_loss(
            data,
            preds,
            self.hparams.opt_inliers_only,
            cls=self.cls,
            rpthres=self.hparams.rpthres,
        )

        # Compute metrics per step
        metrics = defaultdict(list)
        raw_pose_errs = compute_metrics_batch(
            metrics, data, preds, cls=self.cls, sc_thres=0.5
        )
        return losses, metrics, raw_pose_errs

    def training_step(self, data, batch_idx):
        if data["name"] is None:
            return None

        # Compute loss and metrics
        losses, metrics, _ = self.compute_loss_and_metrics(data)
        loss = losses["loss"]

        # Log metrics
        log_metrics = {f"train/{k}": v for k, v in metrics.items()}
        log_metrics.update({f"train/{k}": v for k, v in losses.items()})
        self.log_dict(log_metrics, on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx):
        if data["name"] is None:
            return None

        # Compute loss and metrics
        losses, metrics, raw_pose_errs = self.compute_loss_and_metrics(data)

        # Log metrics
        log_metrics = {f"val/{k}": v for k, v in metrics.items()}
        log_metrics.update({f"val/{k}": v for k, v in losses.items()})
        self.log_dict(log_metrics)
        return raw_pose_errs

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        if len(outputs) == 0:
            return
        metrics = {}
        quantile_tags = ["q1", "q2", "q3"]
        quantile_ratios = torch.tensor([0.25, 0.5, 0.75])
        for k in ("R_err", "t_err"):
            errors = torch.cat([item[k] for item in outputs])
            errors_ = errors[errors > -1]
            quantile_errs = (
                torch.quantile(errors_, quantile_ratios)
                if len(errors_) > 0
                else [-1.0] * 3
            )
            for err, tag in zip(quantile_errs, quantile_tags):
                metrics[f"{k}_{tag}"] = err
        metrics["failed"] = torch.sum(errors == -1).item()
        self.logger.log_hyperparams(params={}, metrics=metrics, step=self.global_step)


def train(args, exp_name):
    pl.seed_everything(args.seed)

    # Define output dir
    if args.overfit:
        args.odir = os.path.join(args.odir, f"overfit{args.overfit}")

    # Logging
    logger = TensorBoardLoggerMetrics(
        args.odir,
        name=exp_name,
        version=args.resume_version,
        default_hp_metric=dict(
            R_err_q1=-1.0,
            t_err_q1=-1.0,
            R_err_q2=-1.0,
            t_err_q2=-1.0,
            R_err_q3=-1.0,
            t_err_q3=-1.0,
            failed=-1.0,
        ),
    )

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1, save_last=True, monitor="val/loss", mode="min", filename="best"
    )
    callbacks = [checkpoint_callback]

    # Training
    last_ckpt = os.path.join(
        args.odir, exp_name, args.resume_version, "checkpoints/last.ckpt"
    )
    if os.path.isfile(last_ckpt):
        """The resuming currently not ideal, need manually specify the resume
        version. If one can assure there's always only one version generated,
        i.e., the output folder uniquely named based on config.
        Then resume can work by hard set to load the version_0 always.
        """
        print(f"# Resuming from {last_ckpt}!!")

        # Initialize the trainer from the last checkpoint
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            deterministic=False,
            callbacks=callbacks,
            resume_from_checkpoint=os.path.join(last_ckpt),
            num_sanity_val_steps=-1,  # ensure when resuming that the metric is computed over the entire validation set
        )
    else:
        # Start from scratch
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            deterministic=False,
            callbacks=callbacks,
        )

    # Init model
    model = MatcherTrainer(args)

    # Initialize data loaders
    train_loader = init_data_loader(args, split=args.train_split)
    val_loader = init_data_loader(
        args, split=args.val_split, outlier_rate=[0, 1], npts=[10, 1024]
    )
    trainer.fit(model, train_loader, val_loader)
