import logging
import os
import time

import matplotlib

matplotlib.use("Agg")  # headless: figures only go to MLflow; Tk breaks forked DataLoader workers
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from monai.data import DataLoader, PersistentDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import PatchAdversarialLoss, PerceptualLoss, SSIMLoss
from monai.networks.nets import PatchDiscriminator
from monai.transforms.utils import allow_missing_keys_mode
from monai.utils import set_determinism
from monai.visualize import matshow3d
from torch.nn import L1Loss
from tqdm import tqdm

from src.constants import TASK_PATHS
from src.data_handling import (
    get_data_paths,
    get_vae_post_transforms,
    get_vae_train_transforms,
    get_vae_val_transforms,
    make_transform_hash,
)
from src.data_handling.datasets import split_data
from src.losses import FFL3D, KLLoss
from src.metrics.image_metrics import ImageMetrics
from src.networks.autoencoder_kl_sigmoid import AutoencoderKLSigmoid
from src.utils.mlflow_utils import log_config, setup_mlflow

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
set_determinism(seed=RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_metrics = ImageMetrics()

        self._setup_logging()
        self._load_config_params()
        self.resume_state = self._load_resume_checkpoint()
        self._setup_mlflow()

        logging.info(f"Trainer initialized with config: {self.config}")

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True,  # MLflow/MONAI install root handlers on import, which turns basicConfig into a no-op
        )

    def _setup_mlflow(self):
        experiment_name = self.config["mlflow"].get("experiment_name", "SSL_Training")
        logging.info(f"Setting up MLflow experiment: {experiment_name}")

        if self.resume_state is not None:
            # Continue the original run; its parameters are already logged and
            # MLflow does not allow overwriting them.
            self.run = setup_mlflow(
                self.config,
                experiment_name=experiment_name,
                run_id=self.resume_state["mlflow_run_id"],
            )
            mlflow.set_tag("resumed_from_epoch", self.resume_state["epoch"] + 1)
        else:
            self.run = setup_mlflow(self.config, experiment_name=experiment_name)
            log_config(self.config)
            mlflow.log_param("device", self.device.type)
            mlflow.log_param("random_seed", RANDOM_SEED)

        self.output_dir = os.path.join("checkpoints", self.run.info.run_id)
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_resume_checkpoint(self):
        if not self.resume_checkpoint_path:
            return None
        if not os.path.exists(self.resume_checkpoint_path):
            raise FileNotFoundError(
                f"resume_checkpoint_path not found: {self.resume_checkpoint_path}"
            )
        logging.info(f"Resuming from checkpoint: {self.resume_checkpoint_path}")
        return torch.load(self.resume_checkpoint_path, map_location="cpu", weights_only=True)

    def _load_config_params(self):
        # Data settings
        self.patch_size = tuple(self.config["data"].get("patch_size", [64, 64, 64]))
        self.train_split = self.config["data"].get("train_split", 0.7)
        self.val_split = self.config["data"].get("val_split", 0.15)
        self.test_split = self.config["data"].get("test_split", 0.15)
        self.tasks = self.config["data"].get("tasks", [1, 2])
        self.num_workers = self.config["data"].get("num_workers", 4)
        self.debug_mode = bool(self.config.get("debug", False))

        # Training settings
        self.batch_size = self.config["training"].get("batch_size", 1)
        self.num_epochs = self.config["training"].get("epochs", 200)
        self.learning_rate = float(self.config["training"].get("learning_rate", 1e-4))

        # Loss weights
        self.pixel_wise_weight = float(self.config["training"].get("pixel_wise_weight", 1.0))
        self.perceptual_weight = float(
            self.config["training"].get("perceptual_weight", 0.001)
        )
        self.ssim_weight = float(self.config["training"].get("ssim_weight", 0.1))
        self.focal_frequency_weight = float(self.config["training"].get("focal_frequency_weight", 1e-4))

        self.adv_weight = float(self.config["training"].get("adv_weight", 0.01))
        self.max_kl_weight = float(self.config["training"].get("max_kl_weight", 1e-2))
        self.kl_annealing_cycles = self.config["training"].get("kl_annealing_cycles", 4)
        self.kl_annealing_ratio = self.config["training"].get("kl_annealing_ratio", 0.5)

        self.warmup_epochs = self.config["training"].get("warmup_epochs", 5)
        self.val_interval = self.config["training"].get("val_interval", 5)
        self.save_interval = self.config["training"].get("save_interval", 20)
        self.test_checkpoint = self.config["training"].get("test_checkpoint", "best")
        if self.test_checkpoint not in ("best", "final"):
            raise ValueError(
                f"training.test_checkpoint must be 'best' or 'final', got '{self.test_checkpoint}'"
            )
        self.resume_checkpoint_path = self.config["training"].get("resume_checkpoint_path")

        # Inference settings
        self.sw_batch_size = self.config["inference"].get("sw_batch_size", 1)
        self.overlap = self.config["inference"].get("overlap", 0.5)
        self.mode = self.config["inference"].get("mode", "gaussian")

    def _prepare_data(self):
        logging.info("Loading data...")
        data_paths = [path for task in self.tasks for path in TASK_PATHS[task]]
        _, cts_paths, masks_paths = get_data_paths(data_paths, debug=self.debug_mode)
        data = [{"image": ct, "mask": mask} for ct, mask in zip(cts_paths, masks_paths)]
        logging.info(f"First data sample: {data[0]}")

        train_data, val_data, test_data = split_data(
            data=data,
            train_split=self.train_split,
            val_split=self.val_split,
            test_split=self.test_split,
            random_seed=RANDOM_SEED,
        )

        logging.info(
            f"Data split: {len(train_data)} train samples, {len(val_data)} validation samples"
        )

        train_ds = PersistentDataset(
            data=train_data,
            transform=get_vae_train_transforms(patch_size=self.patch_size),
            cache_dir="cache/train",
            hash_transform=make_transform_hash(self.patch_size),
        )

        val_ds = PersistentDataset(
            data=val_data,
            transform=get_vae_val_transforms(patch_size=self.patch_size),
            cache_dir="cache/val",
            hash_transform=make_transform_hash(self.patch_size),
        )

        test_ds = PersistentDataset(
            data=test_data,
            transform=get_vae_val_transforms(patch_size=self.patch_size),
            cache_dir="cache/test",
            hash_transform=make_transform_hash(self.patch_size),
        )

        self.val_transforms = val_ds.transform

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        if self.resume_state is None:
            mlflow.log_param("dataset_size", len(data))
            mlflow.log_param("train_dataset_size", len(train_data))
            mlflow.log_param("val_dataset_size", len(val_data))
            mlflow.log_param("test_dataset_size", len(test_data))

    def _setup_models(self):
        self.autoencoder = AutoencoderKLSigmoid(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 64),
            latent_channels=3,
            num_res_blocks=1,
            norm_num_groups=16,
            attention_levels=(False, False, True),
        )
        self.autoencoder.to(self.device)

        self.discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3,
            channels=32,
            in_channels=1,  # autoencoder input
            out_channels=1,
            norm="INSTANCE"
        )
        self.discriminator.to(self.device)

        if self.resume_state is None:
            mlflow.log_param("autoencoder_name", self.autoencoder.__class__.__name__)
            mlflow.log_param("autoencoder_architecture_summary", str(self.autoencoder))
            mlflow.log_param("discriminator_name", self.discriminator.__class__.__name__)
            mlflow.log_param("discriminator_architecture_summary", str(self.discriminator))

    def _setup_losses(self):
        self.l1_loss = L1Loss().to(self.device)
        self.loss_perceptual = PerceptualLoss(
            spatial_dims=3,
            network_type="radimagenet_resnet50",
            is_fake_3d=True,
            fake_3d_ratio=0.25,
        ).to(self.device)
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0, win_size=7).to(self.device)
        self.focal_frequency_loss = FFL3D().to(self.device)
        self.adv_loss = PatchAdversarialLoss(criterion="hinge").to(self.device)
        self.kl_loss = KLLoss().to(self.device)

    def _setup_optimizers(self):
        self.optimizer_g = torch.optim.Adam(
            params=self.autoencoder.parameters(), lr=self.learning_rate
        )
        self.optimizer_d = torch.optim.Adam(
            params=self.discriminator.parameters(), lr=5e-5
        )
        self.scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_g,
            mode="min",
            factor=0.1,
            patience=5,
            min_lr=1e-7,
            threshold=1e-3,
        )

    def _train_generator_step(self, images, epoch):
        self.optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = self.autoencoder(images)

        losses = self._reconstruction_losses(reconstruction, images)
        kl_loss = self.kl_loss(z_mu, z_sigma)

        loss_g_base = losses["total"]
        total_no_kl = loss_g_base.item()

        loss_g_base += self.kl_weight * kl_loss

        loss_g_adv = 0.0
        if epoch >= self.warmup_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            loss_g_adv = self.adv_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
            loss_g = loss_g_base + self.adv_weight * loss_g_adv
        else:
            loss_g = loss_g_base

        loss_g.backward()
        self.optimizer_g.step()

        return {
            "total": loss_g.item(),
            "total_base": total_no_kl,
            "kl": kl_loss.item(),
            "adversarial": (
                loss_g_adv.item()
                if isinstance(loss_g_adv, torch.Tensor)
                else loss_g_adv
            ),
            "reconstruction_tensor": reconstruction,
            "pixel-wise": losses["pixel_wise"].item(),
            "perceptual": self._item(losses["perceptual"]),
            "ssim": self._item(losses["ssim"]),
            "focal_frequency": self._item(losses["focal_frequency"]),
        }

    def _reconstruction_losses(self, outputs, images):
        """
        Weighted reconstruction loss and its terms. Terms with weight 0 are skipped
        (returned as 0.0), except L1: it is always computed because it is the
        model-selection metric and it keeps the total a tensor.
        """
        losses = {
            "pixel_wise": self.l1_loss(outputs, images),
            "perceptual": 0.0,
            "ssim": 0.0,
            "focal_frequency": 0.0,
        }
        if self.perceptual_weight > 0:
            losses["perceptual"] = self.loss_perceptual(outputs.float(), images.float())
        if self.ssim_weight > 0:
            losses["ssim"] = self.ssim_loss(outputs, images)
        if self.focal_frequency_weight > 0:
            losses["focal_frequency"] = self.focal_frequency_loss(outputs, images)

        losses["total"] = (
            self.pixel_wise_weight * losses["pixel_wise"]
            + self.perceptual_weight * losses["perceptual"]
            + self.ssim_weight * losses["ssim"]
            + self.focal_frequency_weight * losses["focal_frequency"]
        )
        return losses

    @staticmethod
    def _item(value):
        return value.item() if isinstance(value, torch.Tensor) else value

    def _infer_and_restore(self, batch, post_tf):
        """Sliding-window reconstruction of a full volume, plus the volume back in HU."""
        images = batch["image"].to(self.device)
        outputs = sliding_window_inference(
            inputs=images,
            roi_size=self.patch_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.autoencoder.reconstruct,
            overlap=self.overlap,
            mode=self.mode,
            device=self.device,
            sw_device=self.device,
        )
        losses = self._reconstruction_losses(outputs, images)

        batch["pred"] = outputs
        with allow_missing_keys_mode(self.val_transforms):
            restored = [post_tf(sample) for sample in decollate_batch(batch)]

        gt = np.squeeze(restored[0]["image"].cpu().numpy())
        recon = np.squeeze(restored[0]["pred"].cpu().numpy())
        mask = np.squeeze(restored[0]["mask"].cpu().numpy())
        scores = self.image_metrics.score_patient(gt_img=gt, synthetic_ct=recon, mask=mask)

        return losses, restored[0], scores

    def _train_discriminator_step(self, reconstruction, images, epoch):
        if epoch < self.warmup_epochs:
            return {"total": 0.0, "fake": 0.0, "real": 0.0}

        self.optimizer_d.zero_grad(set_to_none=True)
        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(
            logits_fake, target_is_real=False, for_discriminator=True
        )

        logits_real = self.discriminator(images.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(
            logits_real, target_is_real=True, for_discriminator=True
        )

        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
        loss_d = self.adv_weight * discriminator_loss

        loss_d.backward()
        self.optimizer_d.step()

        return {
            "total": loss_d.item(),
            "fake": loss_d_fake.item(),
            "real": loss_d_real.item(),
        }

    def _validate_epoch(self, epoch):
        self.autoencoder.eval()
        val_step = 0
        val_losses = {
            "total_g": 0, "pixel_wise": 0, "perceptual": 0,
            "ssim": 0, "focal_frequency": 0,
        }

        val_metrics = {"mae": [], "psnr": [], "ms_ssim": []}
        val_bar = tqdm(
            total=len(self.val_loader), desc="Validation", unit="step", leave=False
        )

        post_tf = get_vae_post_transforms(self.val_transforms)

        with torch.no_grad():
            for val_batch in self.val_loader:
                val_step += 1
                losses, restored, scores = self._infer_and_restore(val_batch, post_tf)

                val_losses["total_g"] += losses["total"].item()
                for key in ("pixel_wise", "perceptual", "ssim", "focal_frequency"):
                    val_losses[key] += self._item(losses[key])

                if self.config["mlflow"].get("log_images", False) and val_step == 1:
                    self._log_validation_images(restored["image"].unsqueeze(0), restored["pred"].unsqueeze(0), epoch)
                    self._log_validation_volume(restored["pred"].unsqueeze(0), epoch)

                tqdm.write(
                    f"Validation Step {val_step} - MAE: {scores['mae']:.4f}, "
                    f"PSNR: {scores['psnr']:.4f}, MS-SSIM: {scores['ms_ssim']:.4f}"
                )

                for key, value in scores.items():
                    val_metrics[key].append(value)

                val_bar.update(1)

        val_bar.close()
        avg_val_losses = {f"val_{k}": v / val_step for k, v in val_losses.items()}
        avg_metrics = {f"val_{k}": np.mean(v) for k, v in val_metrics.items()}

        mlflow.log_metrics(avg_val_losses, step=epoch)
        mlflow.log_metrics(avg_metrics, step=epoch)

        losses_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_val_losses.items()])
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        tqdm.write(
            f"  Validation - Avg Losses: {losses_str}, Avg Metrics: {metrics_str}"
        )

        # L1 on the normalized volume: comparable across experiments whatever their loss weights
        avg_val_loss = avg_val_losses["val_pixel_wise"]
        return avg_val_loss

    def _test_epoch(self):
        self.autoencoder.eval()
        test_step = 0
        test_losses = {
            "total_g": 0, "pixel_wise": 0, "perceptual": 0,
            "ssim": 0, "focal_frequency": 0,
        }
        test_metrics = {"mae": [], "psnr": [], "ms_ssim": []}
        test_bar = tqdm(
            total=len(self.test_loader), desc="Testing", unit="step", leave=False
        )

        post_tf = get_vae_post_transforms(self.val_transforms)

        with torch.no_grad():
            for test_batch in self.test_loader:
                test_step += 1
                losses, restored, scores = self._infer_and_restore(test_batch, post_tf)

                test_losses["total_g"] += losses["total"].item()
                for key in ("pixel_wise", "perceptual", "ssim", "focal_frequency"):
                    test_losses[key] += self._item(losses[key])

                self._log_validation_images(restored["image"].unsqueeze(0), restored["pred"].unsqueeze(0), epoch=test_step, test=True)
                self._log_validation_volume(restored["image"].unsqueeze(0), epoch=test_step, test=True, test_original=True)
                self._log_validation_volume(restored["pred"].unsqueeze(0), epoch=test_step, test=True, test_original=False)

                tqdm.write(
                    f"Test Step {test_step} - MAE: {scores['mae']:.4f}, "
                    f"PSNR: {scores['psnr']:.4f}, MS-SSIM: {scores['ms_ssim']:.4f}"
                )
                mlflow.log_metrics({
                    f"test_step_{test_step}_mae": scores["mae"],
                    f"test_step_{test_step}_psnr": scores["psnr"],
                    f"test_step_{test_step}_ms_ssim": scores["ms_ssim"],
                    f"test_step_{test_step}_pixel_wise_loss": losses["pixel_wise"].item(),
                    f"test_step_{test_step}_perceptual_loss": self._item(losses["perceptual"]),
                    f"test_step_{test_step}_ssim_loss": self._item(losses["ssim"]),
                    f"test_step_{test_step}_focal_frequency_loss": self._item(losses["focal_frequency"]),
                    f"test_step_{test_step}_total_generator_loss": losses["total"].item(),
                }, step=0)

                for key, value in scores.items():
                    test_metrics[key].append(value)

                test_bar.update(1)

        test_bar.close()
        avg_test_losses = {k: v / test_step for k, v in test_losses.items()}
        avg_metrics = {f"test_{k}": np.mean(v) for k, v in test_metrics.items()}

        mlflow.log_metrics({"test_" + k: v for k, v in avg_test_losses.items()}, step=0)
        mlflow.log_metrics(avg_metrics, step=0)

        losses_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_test_losses.items()])
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        tqdm.write(
            f"  Test - Avg Losses: {losses_str}, Avg Metrics: {metrics_str}"
        )

        return avg_test_losses, avg_metrics

    def _log_validation_images(self, val_images, val_outputs, epoch, test=False):
        img_slice = val_images[0, 0, :, :, val_images.shape[-1] // 2].cpu().numpy()
        recon_slice = (
            val_outputs[0, 0, :, :, val_outputs.shape[-1] // 2].cpu().numpy()
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_slice, cmap="gray")
        axes[0].set_title("Original Image Slice")
        axes[0].axis("off")
        axes[1].imshow(recon_slice, cmap="gray")
        axes[1].set_title("Reconstruction Slice")
        axes[1].axis("off")
        plt.tight_layout()
        if not test:
            file_name = f"validation_image_epoch_{epoch+1}.png"
        else:
            file_name = f"test_image_epoch_{epoch}.png"
        mlflow.log_figure(fig, file_name)
        plt.close(fig)

    def _log_validation_volume(self, val_outputs, epoch=None, test=False, test_original=False):
        if not test:
            title = "Validation Reconstruction Volume" if epoch is not None else "Original Volume"
            artifact_name = f"validation_volume_epoch_{epoch+1}.png" if epoch is not None else "original_volume.png"
        else:
            title = "Test Reconstruction Volume"
            artifact_name = f"test_reconstruction_volume_epoch_{epoch}.png" if not test_original else f"test_original_volume_epoch_{epoch}.png"

        fig, _ = matshow3d(
            volume=val_outputs[0].cpu().numpy(),
            title=title,
            figsize=(50, 50),
            every_n=10,
            frame_dim=-1,
            cmap="gray",
            show=False,
        )
        plt.tight_layout()
        mlflow.log_figure(fig, artifact_name)
        plt.close(fig)

    def _update_kl_weight(self, epoch):
        if self.kl_annealing_cycles == 0:
            self.kl_weight = self.max_kl_weight
            return

        period = self.num_epochs / self.kl_annealing_cycles
        epoch_in_cycle = epoch % period
        tau = epoch_in_cycle / (period * self.kl_annealing_ratio)

        if tau <= 1.0:
            self.kl_weight = float(self.max_kl_weight * tau)
        else:
            self.kl_weight = float(self.max_kl_weight)

    def _save_best_models(self, best_val_metric, epoch):
        mlflow.log_metric("best_val_recon_loss_epoch", epoch)
        mlflow.log_metric("best_val_recon_loss", best_val_metric)

        self.best_autoencoder_path = os.path.join(self.output_dir, "best_autoencoder_model.pth")
        best_discriminator_path = os.path.join(self.output_dir, "best_discriminator_model.pth")

        torch.save(self.autoencoder.state_dict(), self.best_autoencoder_path)
        torch.save(self.discriminator.state_dict(), best_discriminator_path)

        mlflow.log_artifact(self.best_autoencoder_path, artifact_path="models")
        mlflow.log_artifact(best_discriminator_path, artifact_path="models")

        tqdm.write(
            f"New best model saved with validation recon loss: {best_val_metric:.4f}"
        )

    def _save_checkpoint(self, epoch, best_val_metric):
        autoencoder_ckpt_path = os.path.join(
            self.output_dir, f"autoencoder_epoch_{epoch+1}.pth"
        )

        discriminator_ckpt_path = os.path.join(
            self.output_dir, f"discriminator_epoch_{epoch+1}.pth"
        )

        torch.save(self.autoencoder.state_dict(), autoencoder_ckpt_path)
        torch.save(self.discriminator.state_dict(), discriminator_ckpt_path)

        mlflow.log_artifact(autoencoder_ckpt_path, artifact_path="checkpoints")
        mlflow.log_artifact(discriminator_ckpt_path, artifact_path="checkpoints")

        # Full training state to resume from (kept locally, overwritten each time)
        torch.save(
            {
                "epoch": epoch,
                "mlflow_run_id": self.run.info.run_id,
                "best_val_metric": best_val_metric,
                "best_autoencoder_path": self.best_autoencoder_path,
                "autoencoder": self.autoencoder.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "optimizer_g": self.optimizer_g.state_dict(),
                "optimizer_d": self.optimizer_d.state_dict(),
                "scheduler_g": self.scheduler_g.state_dict(),
            },
            os.path.join(self.output_dir, "latest_checkpoint.pth"),
        )

        tqdm.write(f"Checkpoint saved at epoch {epoch+1}")

    def _restore_training_state(self):
        """Load models, optimizers and scheduler from the resume checkpoint. Returns the start epoch."""
        if self.resume_state is None:
            return 0

        state = self.resume_state
        self.autoencoder.load_state_dict(state["autoencoder"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer_g.load_state_dict(state["optimizer_g"])
        self.optimizer_d.load_state_dict(state["optimizer_d"])
        self.scheduler_g.load_state_dict(state["scheduler_g"])
        self.best_autoencoder_path = state["best_autoencoder_path"]

        start_epoch = state["epoch"] + 1
        logging.info(f"Resumed training state, continuing at epoch {start_epoch + 1}")
        return start_epoch

    def train(self):
        self._prepare_data()
        self._setup_models()
        self._setup_losses()
        self._setup_optimizers()

        best_val_metric = float("inf")
        self.best_autoencoder_path = None
        start_epoch = self._restore_training_state()
        if self.resume_state is not None:
            best_val_metric = self.resume_state["best_val_metric"]

        # Log first original volume with self._log_validation_volume to compare with validation outputs
        if self.config["mlflow"].get("log_original_volume", False) and start_epoch == 0:
            original_volume = self.val_loader.dataset[0]["image"]
            self._log_validation_volume(original_volume, epoch=None)

        logging.info("Starting training...")

        self.kl_weight = 0.0  # Start with 0

        for epoch in range(start_epoch, self.num_epochs):
            self.autoencoder.train()
            self.discriminator.train()
            epoch_start = time.time()

            self._update_kl_weight(epoch)

            epoch_losses = {
                "g_loss": 0, "d_loss": 0, "pixel_wise": 0,
                "perceptual": 0, "ssim": 0, "focal_frequency": 0, "kl": 0,
                "adv_g": 0, "adv_d_fake": 0, "adv_d_real": 0
            }
            step = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                unit="step",
                leave=True,
            )

            for batch_data in progress_bar:
                step += 1
                images = batch_data["image"].to(self.device)

                g_losses = self._train_generator_step(images, epoch)
                d_losses = self._train_discriminator_step(
                    g_losses["reconstruction_tensor"], images, epoch
                )

                epoch_losses["g_loss"] += g_losses["total"]
                epoch_losses["pixel_wise"] += g_losses["pixel-wise"]
                epoch_losses["perceptual"] += g_losses["perceptual"]
                epoch_losses["ssim"] += g_losses["ssim"]
                epoch_losses["focal_frequency"] += g_losses["focal_frequency"]
                epoch_losses["kl"] += g_losses["kl"]
                epoch_losses["adv_g"] += g_losses["adversarial"]
                epoch_losses["d_loss"] += d_losses["total"]
                epoch_losses["adv_d_fake"] += d_losses["fake"]
                epoch_losses["adv_d_real"] += d_losses["real"]

                progress_bar.set_postfix(
                    {
                        "G_loss": f"{g_losses['total']:.4f}",
                        "D_loss": f"{d_losses['total']:.4f}",
                        "Pixel-wise": f"{g_losses['pixel-wise']:.4f}",
                        "Perceptual": f"{g_losses['perceptual']:.4f}",
                        "SSIM": f"{g_losses['ssim']:.4f}",
                        "Focal Frequency": f"{g_losses['focal_frequency']:.4f}",
                        "KL": f"{g_losses['kl']:.4f}",
                    }
                )

            progress_bar.close()
            epoch_time = time.time() - epoch_start

            # Log average training losses
            avg_losses = {k: v / step for k, v in epoch_losses.items()}
            train_metrics = {
                "train_generator_loss": avg_losses["g_loss"],
                "train_discriminator_loss": avg_losses["d_loss"],
                "train_pixel_wise_loss": avg_losses["pixel_wise"],
                "train_kl_loss": avg_losses["kl"],
                "train_perceptual_loss": avg_losses["perceptual"],
                "train_ssim_loss": avg_losses["ssim"],
                "train_focal_frequency_loss": avg_losses["focal_frequency"],
                "kl_weight": self.kl_weight,
                "train_adv_generator_loss": avg_losses["adv_g"],
                "train_adv_discriminator_fake_loss": avg_losses["adv_d_fake"],
                "train_adv_discriminator_real_loss": avg_losses["adv_d_real"],
                "epoch_time_seconds": epoch_time,
            }
            mlflow.log_metrics(train_metrics, step=epoch)

            tqdm.write(f"Epoch {epoch + 1}/{self.num_epochs} Summary:")
            tqdm.write(
                f"  Avg Train G Loss: {avg_losses['g_loss']:.4f}, Avg Train D Loss: {avg_losses['d_loss']:.4f}"
            )
            tqdm.write(
                f"  Avg Pixel-wise Loss: {avg_losses['pixel_wise']:.4f}, Avg Perceptual Loss: {avg_losses['perceptual']:.4f}, Avg SSIM Loss: {avg_losses['ssim']:.4f}, Avg KL Loss: {avg_losses['kl']:.4f}"
            )
            tqdm.write(f"  Time: {epoch_time:.2f}s")

            # Validation step
            if (epoch + 1) % self.val_interval == 0:
                val_epoch_start = time.time()
                avg_val_recon_loss = self._validate_epoch(epoch)
                val_epoch_time = time.time() - val_epoch_start

                mlflow.log_metrics({
                    "learning_rate_g": self.scheduler_g.optimizer.param_groups[0]['lr'],
                    "val_epoch_time": val_epoch_time
                }, step=epoch)

                self.scheduler_g.step(avg_val_recon_loss)

                if avg_val_recon_loss < best_val_metric:
                    best_val_metric = avg_val_recon_loss
                    self._save_best_models(best_val_metric, epoch)

            # Save checkpoint every save_interval epochs
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch, best_val_metric)

        # End of Training
        tqdm.write("Saving final models...")
        mlflow.pytorch.log_model(self.autoencoder, name="final_autoencoder_model", serialization_format="pickle")
        mlflow.pytorch.log_model(self.discriminator, name="final_discriminator_model", serialization_format="pickle")

        final_autoencoder_path = os.path.join(self.output_dir, "final_autoencoder_model.pth")
        final_discriminator_path = os.path.join(self.output_dir, "final_discriminator_model.pth")

        torch.save(self.autoencoder.state_dict(), final_autoencoder_path)
        torch.save(self.discriminator.state_dict(), final_discriminator_path)

        mlflow.log_artifact(final_autoencoder_path, artifact_path="models")
        mlflow.log_artifact(final_discriminator_path, artifact_path="models")

        mlflow.log_metric("final_epoch", self.num_epochs)
        mlflow.log_metric("final_val_recon_loss", best_val_metric)

        # Test
        if self.test_checkpoint == "best":
            if self.best_autoencoder_path is not None:
                tqdm.write(f"Testing best model: {self.best_autoencoder_path}")
                self.autoencoder.load_state_dict(
                    torch.load(self.best_autoencoder_path, map_location=self.device, weights_only=True)
                )
            else:
                logging.warning("No validation was run, so there is no best model: testing the final model")
        mlflow.set_tag("test_checkpoint", self.test_checkpoint if self.best_autoencoder_path else "final")

        test_losses, test_metrics = self._test_epoch()

        str_losses = ", ".join([f"{k}: {v:.4f}" for k, v in test_losses.items()])
        str_metrics = ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])

        tqdm.write(f"Final Test Results - Losses: {str_losses}")
        tqdm.write(f"Final Test Results - Metrics: {str_metrics}")
        for key, value in test_metrics.items():
            tqdm.write(f"Test {key}: {value:.4f}")

        mlflow.end_run()
        logging.info("Training finished.")
