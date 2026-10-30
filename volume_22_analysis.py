import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import matplotlib.gridspec as gridspec
from pathlib import Path

from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms.utils import allow_missing_keys_mode
from monai.data import decollate_batch

from src.networks.autoencoder_kl_sigmoid import AutoencoderKLSigmoid
from src.constants import ALL_TASKS
from src.data_handling import (
    get_data_paths,
    get_vae_val_transforms,
    get_vae_post_transforms,
)
from src.data_handling.datasets import split_data


class DirectVolume22Analyzer:
    """Analyzer that directly loads models from filesystem without MLflow tracking"""

    def __init__(self, experiments_config, volume_index=22):
        """
        Initialize analyzer for specific volume and experiments.

        Args:
            experiments_config: Dict with experiment info:
                {
                    'experiment_1': {'model_path': 'path/to/model.pth', 'name': 'Baseline', 'config': {...}},
                    'experiment_4': {'model_path': 'path/to/model.pth', 'name': 'Worst', 'config': {...}},
                    'experiment_5': {'model_path': 'path/to/model.pth', 'name': 'Best', 'config': {...}}
                }
            volume_index: Index of the test volume to analyze (default: 22)
        """
        self.experiments_config = experiments_config
        self.volume_index = volume_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_data = {}
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def _load_model(self, exp_key, exp_config):
        """Load model directly from filesystem"""
        model_path = exp_config['model_path']

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

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

        logging.info(f"Loading model from: {model_path}")
        autoencoder.load_state_dict(torch.load(model_path, map_location=self.device))
        autoencoder.to(self.device)
        autoencoder.eval()

        return autoencoder, exp_config.get('config', self._get_default_config())

    def _get_default_config(self):
        """Get default configuration"""
        return {
            "data": {
                "patch_size": [64, 64, 64],
                "train_split": 0.7,
                "val_split": 0.10,
                "test_split": 0.20,
                "task1": True
            },
            "inference": {
                "sw_batch_size": 16,
                "overlap": 0.75,
                "mode": "gaussian"
            },
            "debug": False
        }

    def _prepare_test_data(self, config):
        """Prepare test data using configuration"""
        logging.info("Loading test data...")

        _, cts_paths, masks_paths = get_data_paths(
            ALL_TASKS, task1=True, debug=False
        )
        data = [{"image": ct, "mask": mask} for ct, mask in zip(cts_paths, masks_paths)]

        train_data, val_data, test_data = split_data(
            data=data,
            train_split=0.7,
            val_split=0.1,
            test_split=0.2,
            random_seed=42,  # Same seed as in training
        )

        # Check if volume_index is valid
        if self.volume_index >= len(test_data):
            raise ValueError(f"Volume index {self.volume_index} is out of range. Test set has {len(test_data)} volumes.")

        # Get specific volume
        specific_volume_data = [test_data[self.volume_index]]

        test_ds = Dataset(
            data=specific_volume_data,
            transform=get_vae_val_transforms(patch_size=(64, 64, 64)),
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )

        return test_loader, test_ds.transform

    def _generate_prediction(self, autoencoder, test_loader, val_transforms, config):
        """Generate prediction for the specific test volume"""
        post_tf = get_vae_post_transforms(val_transforms)

        with torch.no_grad():
            # Get the test sample
            test_batch = next(iter(test_loader))
            test_images = test_batch["image"].to(self.device)

            # Generate prediction using sliding window inference
            test_outputs = sliding_window_inference(
                inputs=test_images,
                roi_size=(64, 64, 64),
                sw_batch_size=16,
                predictor=autoencoder.reconstruct,
                overlap=0.75,
                mode="gaussian",
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

    def _process_experiment(self, exp_key, exp_config):
        """Process a single experiment"""
        logging.info(f"Processing experiment {exp_key}...")

        autoencoder, config = self._load_model(exp_key, exp_config)
        test_loader, val_transforms = self._prepare_test_data(config)
        original_slice, predicted_slice = self._generate_prediction(autoencoder, test_loader, val_transforms, config)

        # Store for comparison
        self.experiment_data[exp_key] = {
            'name': exp_config['name'],
            'original': original_slice,
            'predicted': predicted_slice
        }

        logging.info(f"Experiment {exp_key} processed successfully")

    def _generate_volume22_figure(self, output_dir):
        """Generate the specific figure for volume 22 with 4 columns and 2 rows"""
        logging.info("Generating Volume 22 specific figure...")

        # Get experiments in order they were processed
        exp_keys = list(self.experiment_data.keys())
        if len(exp_keys) < 3:
            raise ValueError(f"Need at least 3 experiments, got {len(exp_keys)}")

        # Get reference original (should be same for all)
        reference_original = list(self.experiment_data.values())[0]['original']

        # Get predictions and calculate differences
        predictions = []
        exp_names = []
        signed_diffs = []

        for exp_key in exp_keys:
            exp_data = self.experiment_data[exp_key]
            predictions.append(exp_data['predicted'])
            exp_names.append(exp_data['name'])
            signed_diffs.append(exp_data['predicted'] - reference_original)

        # Calculate global intensity range for consistency
        all_images = [reference_original] + predictions
        vmin_global, vmax_global = np.percentile(np.concatenate([img.flatten() for img in all_images]), [1, 99])

        # Calculate global difference range
        all_diffs = np.concatenate([diff.flatten() for diff in signed_diffs])
        diff_max_global = max(abs(np.percentile(all_diffs, 1)), abs(np.percentile(all_diffs, 99)))

        # Create the figure: 4 columns (Original + experiments), 2 rows (Images + Signed differences)
        n_exp = len(exp_keys)
        fig, axes = plt.subplots(2, n_exp + 1, figsize=(6 * (n_exp + 1), 12), constrained_layout=True)

        # Row 1: Images
        # Original image (first column)
        im_orig = axes[0, 0].imshow(reference_original, cmap='gray', vmin=vmin_global, vmax=vmax_global)
        axes[0, 0].set_title(f'Volumen\noriginal', fontsize=18, fontweight='bold')
        axes[0, 0].axis('off')

        # Predictions
        for i, (exp_name, prediction) in enumerate(zip(exp_names, predictions)):
            im_pred = axes[0, i+1].imshow(prediction, cmap='gray', vmin=vmin_global, vmax=vmax_global)
            axes[0, i+1].set_title(f'{exp_name}\nReconstrucción', fontsize=18)
            axes[0, i+1].axis('off')

        # Row 2: Signed differences
        # Empty space under original
        axes[1, 0].text(0.5, 0.5, 'Referencia\n(sin diferencia)',
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=12, style='italic', color='gray')
        axes[1, 0].axis('off')

        # Signed differences
        for i, (exp_name, signed_diff) in enumerate(zip(exp_names, signed_diffs)):
            im_diff = axes[1, i+1].imshow(signed_diff, cmap='RdBu_r', vmin=-diff_max_global, vmax=diff_max_global)
            axes[1, i+1].set_title(f'{exp_name}\nDiferencia con signo', fontsize=18)
            axes[1, i+1].axis('off')

        # Add colorbars
        cbar1 = fig.colorbar(im_orig, ax=axes[0, :], shrink=0.6, aspect=30, label='Intensidad', location='right', pad=0.02)
        cbar2 = fig.colorbar(im_diff, ax=axes[1, 1:], shrink=0.6, aspect=30, label='Diferencia con signo (Rojo=Positivo, Azul=Negativo)', location='right', pad=0.02)

        # Add main title
        fig.suptitle(f'Original vs Reconstrucción y Diferencia con signo', fontsize=16, fontweight='bold')

        # Save the figure
        output_path = os.path.join(output_dir, f"volume_{self.volume_index}_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Volume {self.volume_index} figure saved to: {output_path}")
        return output_path

    def run(self, output_dir="volume22_analysis"):
        """Main execution method"""
        logging.info(f"Starting Volume {self.volume_index} analysis...")
        logging.info(f"Device: {self.device}")

        os.makedirs(output_dir, exist_ok=True)

        # Process each experiment
        for exp_key, exp_config in self.experiments_config.items():
            try:
                self._process_experiment(exp_key, exp_config)
            except Exception as e:
                logging.error(f"Failed to process experiment {exp_key}: {str(e)}")
                continue

        # Generate the specific figure
        if len(self.experiment_data) >= 1:
            figure_path = self._generate_volume22_figure(output_dir)
            logging.info("Analysis completed successfully!")
            return figure_path
        else:
            logging.error("No experiments processed successfully!")
            return None


def main():
    # Define your experiments configuration - UPDATE THESE PATHS!
    experiments_config = {
        'experiment_1': {
            'model_path': 'mlruns/mlruns_1/313770645939694569/7a56cdd2aecc410eac749071cdd68308/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 (Línea Base)'
        },
        'experiment_2': {
            'model_path': 'mlruns/mlruns_2/675918440420340658/48d40c1d5de64fd59dddb31484454246/artifacts/models/best_autoencoder_model.pth',
            'name': 'Perceptual (Caso 2)'
        },
        'experiment_3': {
            'model_path': 'mlruns/mlruns_3/360562246159054027/725589bfcea14434a5ea4cb2a0c27dc3/artifacts/models/best_autoencoder_model.pth',
            'name': 'SSIM (Caso 3)'
        },
        'experiment_4': {
            'model_path': 'mlruns/mlruns_4/702153582262418679/313e751960b44e48a74ca829ea6e66a1/artifacts/models/best_autoencoder_model.pth',
            'name': 'FFL (Caso 4, Peor)'
        },
        'experiment_5': {
            'model_path': 'mlruns/mlruns_5/239927948431294758/ec5ae7551fff4204a529b7c7259fab65/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + Perceptual (Caso 5, Mejor)'
        },
        'experiment_6': {
            'model_path': 'mlruns/mlruns_6/385784109692985319/f8de3076331c4e9fb5e3187ae54a7b2b/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + SSIM (Caso 6)'
        },
        'experiment_7': {
            'model_path': 'mlruns/mlruns_7/744827778112609234/7c8722fe646e465593e5028ff63ed29a/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + FFL (Caso 7)'
        },
        'experiment_8': {
            'model_path': 'mlruns/mlruns_8/148501807127053673/c3d6cbcff4ed4cb39e5bf691407d12df/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + SSIM + FFL (Caso 8)'
        },
        'experiment_9': {
            'model_path': 'mlruns/mlruns_9/346230370302623680/fda24d074c4c4d909160dc5e0c37c399/artifacts/models/best_autoencoder_model.pth',
            'name': 'Perceptual + SSIM + FFL (Caso 9)'
        },
        'experiment_10': {
            'model_path': 'mlruns/mlruns_10/819515919217557588/24198dcbfdbe49ccb4d554dd3751286f/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + Perceptual + SSIM + FFL (Caso 10)'
        }
    }

    # select only experiments 1, 4, and 5 for the specific figure
    experiments_config = {k: v for k, v in experiments_config.items() if k in ['experiment_1', 'experiment_4', 'experiment_5']}

    parser = argparse.ArgumentParser(description="Generate Volume 22 comparison figure")
    parser.add_argument("--volume_index", type=int, default=22,
                       help="Index of the test volume to analyze (default: 22)")
    parser.add_argument("--output_dir", type=str, default="volume22_analysis",
                       help="Output directory for analysis results")

    args = parser.parse_args()

    try:
        analyzer = DirectVolume22Analyzer(
            experiments_config=experiments_config,
            volume_index=args.volume_index
        )
        figure_path = analyzer.run(output_dir=args.output_dir)

        if figure_path:
            print(f"Analysis completed! Figure saved at: {figure_path}")
        else:
            print("Analysis failed!")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
