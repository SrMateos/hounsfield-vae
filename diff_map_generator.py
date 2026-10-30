import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import mlflow
import mlflow.pytorch
from monai.data import PersistentDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms.utils import allow_missing_keys_mode
from monai.data import decollate_batch
from pathlib import Path
import yaml
import matplotlib.gridspec as gridspec

from src.networks.autoencoder_kl_sigmoid import AutoencoderKLSigmoid
from src.constants import ALL_TASKS
from src.data_handling import (
    get_data_paths,
    get_vae_val_transforms,
    get_vae_post_transforms,
    make_transform_hash,
)
from src.data_handling.datasets import split_data
from src.utils.mlflow_utils import load_run_config


class MultiExperimentDifferenceMapGenerator:
    def __init__(self, experiment_ids, run_ids=None, mlruns_path="mlruns"):
        """
        Initialize the multi-experiment difference map generator.

        Args:
            experiment_ids: List of MLflow experiment IDs or single experiment ID
            run_ids: List of specific run IDs (optional, will use best runs if None)
            mlruns_path: Path to MLflow runs directory
        """
        self.experiment_ids = experiment_ids if isinstance(experiment_ids, list) else [experiment_ids]
        self.run_ids = run_ids if run_ids is not None else [None] * len(self.experiment_ids)
        self.mlruns_path = mlruns_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.experiment_data = {}  # Store data for global comparison

        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _find_best_run(self, experiment_id):
        """Find the run with the best validation reconstruction loss in the experiment"""
        client = mlflow.tracking.MlflowClient()

        # Get all runs from the experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["metrics.best_val_recon_loss ASC"],
            max_results=1
        )

        if not runs:
            raise ValueError(f"No runs found in experiment {experiment_id}")

        best_run = runs[0]
        logging.info(f"Found best run: {best_run.info.run_id} with val loss: {best_run.data.metrics.get('best_val_recon_loss', 'N/A')}")
        return best_run.info.run_id

    def _get_experiment_name(self, experiment_id):
        """Get experiment name from MLflow"""
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment(experiment_id)
            return experiment.name
        except:
            return f"experiment_{experiment_id}"

    def _load_config_from_run(self, run_id):
        """Load configuration from MLflow run"""
        return load_run_config(run_id)

    def _load_model(self, experiment_id, run_id):
        """Load model for a specific experiment"""
        if run_id is None:
            run_id = self._find_best_run(experiment_id)

        # Load configuration
        config = self._load_config_from_run(run_id)

        # Initialize model architecture
        autoencoder = AutoencoderKLSigmoid(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 64),
            latent_channels=3,
            num_res_blocks=1,
            norm_num_groups=16,
            attention_levels=(False, False, True),
        )

        # Load model weights
        model_path = os.path.join(
            self.mlruns_path,
            experiment_id,
            run_id,
            "artifacts",
            "models",
            "best_autoencoder_model.pth"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Best model not found at {model_path}")

        logging.info(f"Loading model from: {model_path}")
        autoencoder.load_state_dict(torch.load(model_path, map_location=self.device))
        autoencoder.to(self.device)
        autoencoder.eval()

        return autoencoder, config, run_id

    def _prepare_test_data(self, config):
        """Prepare test data using configuration"""
        logging.info("Loading test data...")

        _, cts_paths, masks_paths = get_data_paths(
            ALL_TASKS, task1=config["data"]["task1"], debug=config["debug"]
        )
        data = [{"image": ct, "mask": mask} for ct, mask in zip(cts_paths, masks_paths)]

        train_data, val_data, test_data = split_data(
            data=data,
            train_split=config["data"]["train_split"],
            val_split=config["data"]["val_split"],
            test_split=config["data"]["test_split"],
            random_seed=42,  # Same seed as in training
        )

        test_ds = PersistentDataset(
            data=test_data,
            transform=get_vae_val_transforms(patch_size=tuple(config["data"]["patch_size"])),
            cache_dir="cache/test_inference",
            hash_transform=make_transform_hash(config["data"]["patch_size"]),
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )

        return test_loader, test_ds.transform

    def _generate_prediction(self, autoencoder, test_loader, val_transforms, config):
        """Generate prediction for first test sample"""
        post_tf = get_vae_post_transforms(val_transforms)

        with torch.no_grad():
            # Get first test sample
            test_batch = next(iter(test_loader))
            test_images = test_batch["image"].to(self.device)

            # Generate prediction using sliding window inference
            test_outputs = sliding_window_inference(
                inputs=test_images,
                roi_size=tuple(config["data"]["patch_size"]),
                sw_batch_size=config["inference"]["sw_batch_size"],
                predictor=autoencoder.reconstruct,
                overlap=config["inference"]["overlap"],
                mode=config["inference"]["mode"],
                device=self.device,
                sw_device=self.device,
            )

            # Prepare batch for post-processing
            test_batch["pred"] = test_outputs
            test_samples = decollate_batch(test_batch)

            # Apply post-transforms to denormalize
            with allow_missing_keys_mode(val_transforms):
                restored = [post_tf(test_sample) for test_sample in test_samples]

            # Extract denormalized tensors and get middle slice
            original_volume = np.squeeze(restored[0]["image"].cpu().numpy())
            predicted_volume = np.squeeze(restored[0]["pred"].cpu().numpy())

            middle_slice_idx = original_volume.shape[-1] // 2
            original_slice = original_volume[:, :, middle_slice_idx]
            predicted_slice = predicted_volume[:, :, middle_slice_idx]

            return original_slice, predicted_slice

    def _process_single_experiment(self, experiment_id, run_id, output_base_dir):
        """Process a single experiment and generate individual maps"""
        logging.info(f"Processing experiment {experiment_id}...")

        # Get experiment name for folder
        experiment_name = self._get_experiment_name(experiment_id)
        safe_name = "".join(c for c in experiment_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        experiment_dir = os.path.join(output_base_dir, f"{safe_name}_{experiment_id}")
        os.makedirs(experiment_dir, exist_ok=True)

        # Load model and generate prediction
        autoencoder, config, actual_run_id = self._load_model(experiment_id, run_id)
        test_loader, val_transforms = self._prepare_test_data(config)
        original_slice, predicted_slice = self._generate_prediction(autoencoder, test_loader, val_transforms, config)

        # Store for global comparison
        self.experiment_data[experiment_id] = {
            'name': safe_name,
            'original': original_slice,
            'predicted': predicted_slice,
            'run_id': actual_run_id
        }

        # Generate individual analysis
        self._generate_individual_analysis(original_slice, predicted_slice, experiment_dir, experiment_name, actual_run_id)

        logging.info(f"Experiment {experiment_id} processed successfully")

    def _generate_individual_analysis(self, original_slice, predicted_slice, output_dir, experiment_name, run_id):
        """Generate analysis for a single experiment"""
        # Calculate different types of difference maps
        abs_difference_slice = np.abs(original_slice - predicted_slice)
        signed_difference_slice = predicted_slice - original_slice

        # Save experiment info
        with open(os.path.join(output_dir, "experiment_info.txt"), 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Generated on: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n\n")

        # Consistent intensity scaling
        vmin_orig, vmax_orig = np.percentile(original_slice, [1, 99])
        vmin_pred, vmax_pred = np.percentile(predicted_slice, [1, 99])
        vmin_common = min(vmin_orig, vmin_pred)
        vmax_common = max(vmax_orig, vmax_pred)

        # Save individual images
        self._save_slice_image(original_slice,
                             os.path.join(output_dir, "original_slice.png"),
                             f"Original Image - {experiment_name}",
                             "gray", vmin_common, vmax_common)

        self._save_slice_image(predicted_slice,
                             os.path.join(output_dir, "predicted_slice.png"),
                             f"Predicted Image - {experiment_name}",
                             "gray", vmin_common, vmax_common)

        self._save_slice_image(abs_difference_slice,
                             os.path.join(output_dir, "absolute_difference_map.png"),
                             f"Absolute Difference - {experiment_name}",
                             "plasma", 0, None)

        diff_max = max(abs(np.percentile(signed_difference_slice, 1)),
                      abs(np.percentile(signed_difference_slice, 99)))
        self._save_slice_image(signed_difference_slice,
                             os.path.join(output_dir, "signed_difference_map.png"),
                             f"Signed Difference - {experiment_name}",
                             "RdBu_r", -diff_max, diff_max)

        # Save combined comparisons
        self._save_comparison_image(original_slice, predicted_slice, abs_difference_slice,
                                  os.path.join(output_dir, "comparison_absolute.png"),
                                  vmin_common, vmax_common)

        self._save_signed_comparison_image(original_slice, predicted_slice, signed_difference_slice,
                                         os.path.join(output_dir, "comparison_signed.png"),
                                         vmin_common, vmax_common, diff_max)

        # Save statistics
        self._save_statistics(original_slice, predicted_slice, abs_difference_slice, signed_difference_slice,
                            os.path.join(output_dir, "statistics.txt"))

    def _generate_global_comparison(self, output_base_dir):
        """Generate global comparison across all experiments"""
        if len(self.experiment_data) < 2:
            logging.warning("Need at least 2 experiments for global comparison")
            return

        logging.info("Generating global comparison...")

        # Get reference original (should be same for all)
        reference_original = list(self.experiment_data.values())[0]['original']

        # Prepare data for plotting
        experiment_names = []
        predictions = []
        signed_diffs = []

        # Calculate global intensity range for consistency
        all_images = [reference_original]
        for exp_data in self.experiment_data.values():
            all_images.append(exp_data['predicted'])

        vmin_global, vmax_global = np.percentile(np.concatenate([img.flatten() for img in all_images]), [1, 99])

        for exp_id, exp_data in self.experiment_data.items():
            experiment_names.append(f"{exp_data['name']}")
            predictions.append(exp_data['predicted'])
            signed_diffs.append(exp_data['predicted'] - reference_original)

        # Calculate global difference range
        all_diffs = np.concatenate([diff.flatten() for diff in signed_diffs])
        diff_max_global = max(abs(np.percentile(all_diffs, 1)), abs(np.percentile(all_diffs, 99)))

        # Create the global comparison plot
        n_experiments = len(self.experiment_data)
        fig, axes = plt.subplots(2, n_experiments + 1, figsize=(6 * (n_experiments + 1), 12),
                                 constrained_layout=True)

        # First row: Original + Predictions
        # Original image (first column)
        im_orig = axes[0, 0].imshow(reference_original, cmap='gray', vmin=vmin_global, vmax=vmax_global)
        axes[0, 0].set_title('Original Image\n(Reference)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Predictions (remaining columns of first row)
        for i, (exp_name, prediction) in enumerate(zip(experiment_names, predictions)):
            im_pred = axes[0, i+1].imshow(prediction, cmap='gray', vmin=vmin_global, vmax=vmax_global)
            axes[0, i+1].set_title(f'Prediction\n{exp_name}', fontsize=10)
            axes[0, i+1].axis('off')

        # Second row: Empty space + Signed differences
        # Empty space under original (no self-difference)
        axes[1, 0].text(0.5, 0.5, 'No self-difference\n(Original vs Original = 0)',
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=12, style='italic', color='gray')
        axes[1, 0].axis('off')

        # Signed differences (remaining columns of second row)
        for i, (exp_name, signed_diff) in enumerate(zip(experiment_names, signed_diffs)):
            im_diff = axes[1, i+1].imshow(signed_diff, cmap='RdBu_r', vmin=-diff_max_global, vmax=diff_max_global)
            axes[1, i+1].set_title(f'Signed Difference\n{exp_name}', fontsize=10) # \n(Red=Over, Blue=Under)
            axes[1, i+1].axis('off')

        # Add colorbars
        # Colorbar for original and predictions (grayscale)
        # fig.colorbar(im_orig, ax=axes[0, :].tolist(), shrink=0.6, aspect=30,
                    # label='Intensity', location='right', pad=0.02)

        # Colorbar for signed differences
        # fig.colorbar(im_diff, ax=axes[1, 1:].tolist(), shrink=0.6, aspect=30,
                    # label='Signed Difference', location='right', pad=0.02)

        cbar1 = fig.colorbar(im_orig, ax=axes[0, :], shrink=0.6, aspect=30, label='Intensity', location='right', pad=0.02)
        cbar2 = fig.colorbar(im_diff, ax=axes[1, 1:], shrink=0.6, aspect=30, label='Signed Difference', location='right', pad=0.02)

        # plt.tight_layout()

        # Save global comparison
        global_comparison_path = os.path.join(output_base_dir, "global_comparison.png")
        plt.savefig(global_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Generate summary statistics comparison
        self._generate_global_statistics(output_base_dir, reference_original)

        logging.info(f"Global comparison saved to: {global_comparison_path}")

    def _generate_global_statistics(self, output_base_dir, reference_original):
        """Generate comparative statistics across all experiments"""
        stats_path = os.path.join(output_base_dir, "global_statistics.txt")

        with open(stats_path, 'w') as f:
            f.write("GLOBAL EXPERIMENT COMPARISON\n")
            f.write("=" * 60 + "\n\n")

            # Header
            f.write("Experiment Comparison Summary:\n")
            f.write("-" * 30 + "\n")
            for exp_id, exp_data in self.experiment_data.items():
                f.write(f"  {exp_data['name']} (ID: {exp_id}, Run: {exp_data['run_id']})\n")
            f.write("\n")

            # Comparative metrics
            f.write("COMPARATIVE METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"{'Experiment':<25} {'MAE':<10} {'RMSE':<10} {'SSIM':<10} {'Max Error':<12}\n")
            f.write("-" * 75 + "\n")

            from skimage.metrics import structural_similarity

            for exp_id, exp_data in self.experiment_data.items():
                predicted = exp_data['predicted']
                abs_diff = np.abs(reference_original - predicted)

                mae = np.mean(abs_diff)
                rmse = np.sqrt(np.mean((reference_original - predicted) ** 2))
                ssim_val = structural_similarity(reference_original, predicted,
                                               data_range=reference_original.max() - reference_original.min())
                max_error = np.max(abs_diff)

                exp_short = exp_data['name'][:20] + "..." if len(exp_data['name']) > 20 else exp_data['name']
                f.write(f"{exp_short:<25} {mae:<10.4f} {rmse:<10.4f} {ssim_val:<10.4f} {max_error:<12.4f}\n")

            f.write("\n")
            f.write("NOTES:\n")
            f.write("- MAE: Mean Absolute Error (lower is better)\n")
            f.write("- RMSE: Root Mean Square Error (lower is better)\n")
            f.write("- SSIM: Structural Similarity Index (higher is better, max=1.0)\n")
            f.write("- Max Error: Maximum absolute difference (lower is better)\n")

        logging.info(f"Global statistics saved to: {stats_path}")

    def _save_slice_image(self, slice_data, filepath, title, colormap, vmin=None, vmax=None):
        """Save a single slice as PNG with proper intensity scaling"""
        # Ensure slice is 2D
        if len(slice_data.shape) > 2:
            slice_data = np.squeeze(slice_data)

        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        im = ax.imshow(slice_data, cmap=colormap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        # Barra de color anclada al eje, sin tight_layout
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        # Evita bbox_inches="tight" para no pelear con constrained_layout
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

    def _save_comparison_image(self, original, predicted, difference, filepath, vmin, vmax):
        original  = np.squeeze(original)
        predicted = np.squeeze(predicted)
        difference= np.squeeze(difference)

        fig = plt.figure(figsize=(20, 6), constrained_layout=True)
        gs  = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1,1,1,0.05])

        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[0,2])
        cax = fig.add_subplot(gs[0,3])  # columna reservada para colorbar

        im1 = ax1.imshow(original,  cmap='gray', vmin=vmin, vmax=vmax);  ax1.set_title('Original');  ax1.axis('off')
        im2 = ax2.imshow(predicted, cmap='gray', vmin=vmin, vmax=vmax);  ax2.set_title('Predicted'); ax2.axis('off')
        im3 = ax3.imshow(difference, cmap='plasma');                     ax3.set_title('|Difference|'); ax3.axis('off')

        # Barra compartida para la imagen de diferencia (puedes cambiar a im1/im2 si prefieres)
        fig.colorbar(im3, cax=cax)
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

    def _save_signed_comparison_image(self, original, predicted, signed_diff, filepath, vmin, vmax, diff_max):
        """Save a comparison image focusing on signed differences"""
        original = np.squeeze(original)
        predicted  = np.squeeze(predicted)
        signed_diff= np.squeeze(signed_diff)

        fig, axes = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True)

        im1 = axes[0,0].imshow(original,  cmap='gray', vmin=vmin, vmax=vmax)
        axes[0,0].set_title('Original Image', fontsize=14);  axes[0,0].axis('off')
        fig.colorbar(im1, ax=axes[0,0], shrink=0.8, pad=0.02)

        im2 = axes[0,1].imshow(predicted, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0,1].set_title('Predicted Image', fontsize=14); axes[0,1].axis('off')
        fig.colorbar(im2, ax=axes[0,1], shrink=0.8, pad=0.02)

        im3 = axes[1,0].imshow(signed_diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        axes[1,0].set_title('Signed Difference', fontsize=14); axes[1,0].axis('off') # \n(Red=Over, Blue=Under)
        fig.colorbar(im3, ax=axes[1,0], shrink=0.8, pad=0.02)

        axes[1,1].imshow(original, cmap='gray', vmin=vmin, vmax=vmax, alpha=0.8)
        axes[1,1].contour(np.abs(signed_diff), levels=5, colors='red', linewidths=1)
        axes[1,1].set_title('Original + Error Contours', fontsize=14); axes[1,1].axis('off')

        fig.savefig(filepath, dpi=300)
        plt.close(fig)

    def _save_statistics(self, original, predicted, abs_difference, signed_difference, filepath):
        """Save statistics about the images and difference"""

        # Calculate additional metrics
        mse = np.mean((original - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(abs_difference)

        # Calculate percentage errors
        original_mean = np.mean(original[original > 0])  # Avoid division by zero
        relative_mae = (mae / original_mean) * 100 if original_mean > 0 else 0

        # Calculate structural similarity
        from skimage.metrics import structural_similarity
        ssim_val = structural_similarity(original, predicted, data_range=original.max() - original.min())

        # Percentile analysis of errors
        error_percentiles = np.percentile(abs_difference, [50, 75, 90, 95, 99])

        stats = {
            'Original Image': {
                'Min': float(original.min()),
                'Max': float(original.max()),
                'Mean': float(original.mean()),
                'Std': float(original.std()),
                'Median': float(np.median(original))
            },
            'Predicted Image': {
                'Min': float(predicted.min()),
                'Max': float(predicted.max()),
                'Mean': float(predicted.mean()),
                'Std': float(predicted.std()),
                'Median': float(np.median(predicted))
            },
            'Absolute Difference': {
                'Min': float(abs_difference.min()),
                'Max': float(abs_difference.max()),
                'Mean (MAE)': float(mae),
                'Std': float(abs_difference.std()),
                'Median': float(np.median(abs_difference)),
                'P50': float(error_percentiles[0]),
                'P75': float(error_percentiles[1]),
                'P90': float(error_percentiles[2]),
                'P95': float(error_percentiles[3]),
                'P99': float(error_percentiles[4])
            },
            'Signed Difference (Pred - Orig)': {
                'Min': float(signed_difference.min()),
                'Max': float(signed_difference.max()),
                'Mean': float(signed_difference.mean()),
                'Std': float(signed_difference.std()),
                'Median': float(np.median(signed_difference))
            },
            'Error Metrics': {
                'MAE (Mean Absolute Error)': float(mae),
                'MSE (Mean Squared Error)': float(mse),
                'RMSE (Root Mean Squared Error)': float(rmse),
                'Relative MAE (%)': float(relative_mae),
                'SSIM (Structural Similarity)': float(ssim_val),
                'Max Absolute Error': float(abs_difference.max()),
                'Pixels with error > 1% of max': int(np.sum(abs_difference > 0.01 * original.max())),
                'Pixels with error > 5% of max': int(np.sum(abs_difference > 0.05 * original.max()))
            }
        }

        with open(filepath, 'w') as f:
            f.write("DETAILED IMAGE ANALYSIS AND ERROR METRICS\n")
            f.write("=" * 60 + "\n\n")

            for category, metrics in stats.items():
                f.write(f"{category}:\n")
                f.write("-" * len(category) + "\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {metric}: {value:.6f}\n")
                    else:
                        f.write(f"  {metric}: {value}\n")
                f.write("\n")

            f.write("INTERPRETATION NOTES:\n")
            f.write("-" * 20 + "\n")
            f.write("• Signed difference: Positive values indicate overprediction, negative underprediction\n")
            f.write("• SSIM closer to 1.0 indicates better structural similarity\n")
            f.write("• Error percentiles show the distribution of prediction errors\n")
            f.write("• Relative MAE shows error as percentage of original image intensity\n")

    def run(self, output_base_dir="multi_experiment_analysis"):
        """Main execution method for multiple experiments"""
        logging.info("Starting multi-experiment analysis...")
        logging.info(f"Processing {len(self.experiment_ids)} experiments")
        logging.info(f"Device: {self.device}")

        os.makedirs(output_base_dir, exist_ok=True)

        # Process each experiment individually
        for i, experiment_id in enumerate(self.experiment_ids):
            run_id = self.run_ids[i] if i < len(self.run_ids) else None
            try:
                self._process_single_experiment(experiment_id, run_id, output_base_dir)
            except Exception as e:
                logging.error(f"Failed to process experiment {experiment_id}: {str(e)}")
                continue

        # Generate global comparison if we have multiple successful experiments
        if len(self.experiment_data) > 1:
            self._generate_global_comparison(output_base_dir)
        elif len(self.experiment_data) == 1:
            logging.warning("Only one experiment processed successfully. Skipping global comparison.")
        else:
            logging.error("No experiments processed successfully!")
            return None

        logging.info("Multi-experiment analysis completed successfully!")
        logging.info(f"Results saved to: {output_base_dir}")

        return self.experiment_data


def main():
    parser = argparse.ArgumentParser(description="Generate difference maps between original and predicted images for multiple experiments")
    parser.add_argument("--experiment_ids", type=str, required=True, nargs='+',
                       help="MLflow experiment IDs (can provide multiple)")
    parser.add_argument("--run_ids", type=str, default=None, nargs='*',
                       help="Specific MLflow run IDs (optional, if not provided, best runs will be selected)")
    parser.add_argument("--mlruns_path", type=str, default="mlruns",
                       help="Path to mlruns directory")
    parser.add_argument("--output_dir", type=str, default="multi_experiment_analysis",
                       help="Output directory for analysis results")

    args = parser.parse_args()

    try:
        generator = MultiExperimentDifferenceMapGenerator(
            experiment_ids=args.experiment_ids,
            run_ids=args.run_ids,
            mlruns_path=args.mlruns_path
        )
        generator.run(output_base_dir=args.output_dir)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
