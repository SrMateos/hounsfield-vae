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


class GridVolumeAnalyzer:
    """Analyzer that creates a 4x4 grid layout for displaying experiments"""

    def __init__(self, experiments_config, volume_index=22):
        """
        Initialize analyzer for specific volume and experiments.

        Args:
            experiments_config: Dict with experiment info
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
            random_seed=42,
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
            test_batch = next(iter(test_loader))
            test_images = test_batch["image"].to(self.device)

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

            test_batch["pred"] = test_outputs
            test_samples = decollate_batch(test_batch)

            with allow_missing_keys_mode(val_transforms):
                restored = [post_tf(test_sample) for test_sample in test_samples]

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

        self.experiment_data[exp_key] = {
            'name': exp_config['name'],
            'original': original_slice,
            'predicted': predicted_slice
        }

        logging.info(f"Experiment {exp_key} processed successfully")

    def _generate_4x4_figure(self, output_dir):
        """Generate single 4x4 grid figure with specific layout"""
        logging.info("Generating 4x4 grid figure...")

        exp_keys = list(self.experiment_data.keys())
        if len(exp_keys) == 0:
            raise ValueError("No experiments processed successfully")

        # Get reference original (should be same for all)
        reference_original = list(self.experiment_data.values())[0]['original']

        # Prepare all experiment data in order
        exp_data_ordered = []
        for exp_key in exp_keys:
            exp_data = self.experiment_data[exp_key]
            signed_diff = exp_data['predicted'] - reference_original
            exp_data_ordered.append({
                'name': exp_data['name'],
                'prediction': exp_data['predicted'],
                'signed_diff': signed_diff
            })

        # Calculate global intensity ranges
        all_images = [reference_original] + [exp['prediction'] for exp in exp_data_ordered]
        vmin_global, vmax_global = np.percentile(np.concatenate([img.flatten() for img in all_images]), [1, 99])

        all_diffs = np.concatenate([exp['signed_diff'].flatten() for exp in exp_data_ordered])
        diff_max_global = max(abs(np.percentile(all_diffs, 1)), abs(np.percentile(all_diffs, 99)))

        # Create 4x4 grid
        fig, axes = plt.subplots(4, 4, figsize=(20, 20), constrained_layout=True)

        # Hide all axes initially
        for i in range(4):
            for j in range(4):
                axes[i, j].axis('off')

        # Row 1: Original + first 3 experiments predictions
        # Position (0,0): Original
        im_orig = axes[0, 0].imshow(reference_original, cmap='gray', vmin=vmin_global, vmax=vmax_global)
        axes[0, 0].set_title('Volumen\noriginal', fontsize=14, fontweight='bold')

        # Positions (0,1), (0,2), (0,3): First 3 experiments
        for i in range(min(3, len(exp_data_ordered))):
            exp = exp_data_ordered[i]
            axes[0, i+1].imshow(exp['prediction'], cmap='gray', vmin=vmin_global, vmax=vmax_global)
            axes[0, i+1].set_title(f"{exp['name']}\nReconstrucción", fontsize=12)

        # Row 2: Difference maps for first 3 experiments
        # Position (1,0): Empty/Label
        axes[1, 0].text(0.5, 0.5, 'Mapas de\ndiferencia con signo',
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=12, fontweight='bold', color='navy')
        axes[1, 0].set_facecolor('lightgray')

        # Positions (1,1), (1,2), (1,3): Difference maps for first 3 experiments
        for i in range(min(3, len(exp_data_ordered))):
            exp = exp_data_ordered[i]
            im_diff = axes[1, i+1].imshow(exp['signed_diff'], cmap='RdBu_r',
                                        vmin=-diff_max_global, vmax=diff_max_global)
            axes[1, i+1].set_title(f"{exp['name']}\nDiferencia", fontsize=12)

        # Row 3: Next 4 experiments predictions (experiments 4-7)
        for i in range(4):
            exp_idx = i + 3  # Start from 4th experiment (index 3)
            if exp_idx < len(exp_data_ordered):
                exp = exp_data_ordered[exp_idx]
                axes[2, i].imshow(exp['prediction'], cmap='gray', vmin=vmin_global, vmax=vmax_global)
                axes[2, i].set_title(f"{exp['name']}\nReconstrucción", fontsize=12)

        # Row 4: Difference maps for next 4 experiments (experiments 4-7)
        for i in range(4):
            exp_idx = i + 3  # Start from 4th experiment (index 3)
            if exp_idx < len(exp_data_ordered):
                exp = exp_data_ordered[exp_idx]
                axes[3, i].imshow(exp['signed_diff'], cmap='RdBu_r',
                                vmin=-diff_max_global, vmax=diff_max_global)
                axes[3, i].set_title(f"{exp['name']}\nDiferencia", fontsize=12)

        # Add colorbars
        # Intensity colorbar for predictions
        cbar1 = fig.colorbar(im_orig, ax=axes[0, :], shrink=0.6, aspect=20,
                           label='Intensidad', location='right', pad=0.02)
        cbar1_3 = fig.colorbar(im_orig, ax=axes[2, :], shrink=0.6, aspect=20,
                             label='Intensidad', location='right', pad=0.02)

        # Difference colorbars
        if len(exp_data_ordered) > 0:
            cbar2 = fig.colorbar(im_diff, ax=axes[1, :], shrink=0.6, aspect=20,
                               label='Diferencia con signo (Rojo=+, Azul=-)',
                               location='right', pad=0.02)
            cbar2_4 = fig.colorbar(im_diff, ax=axes[3, :], shrink=0.6, aspect=20,
                                 label='Diferencia con signo (Rojo=+, Azul=-)',
                                 location='right', pad=0.02)

        # Add main title
        fig.suptitle(f'Análisis del Volumen {self.volume_index}', fontsize=16, fontweight='bold')

        # Save figure
        output_path = os.path.join(output_dir, f"volume_{self.volume_index}_grid.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Grid figure saved to: {output_path}")
        return output_path

    def run(self, output_dir="volume_grid_analysis"):
        """Main execution method"""
        logging.info(f"Starting Volume {self.volume_index} grid analysis...")
        logging.info(f"Device: {self.device}")

        os.makedirs(output_dir, exist_ok=True)

        # Process each experiment
        for exp_key, exp_config in self.experiments_config.items():
            try:
                self._process_experiment(exp_key, exp_config)
            except Exception as e:
                logging.error(f"Failed to process experiment {exp_key}: {str(e)}")
                continue

        # Generate single grid figure
        if len(self.experiment_data) >= 1:
            figure_path = self._generate_4x4_figure(output_dir)
            logging.info("Grid analysis completed successfully!")
            return figure_path
        else:
            logging.error("No experiments processed successfully!")
            return None


def main():
    # Define all your experiments configuration in the order you want them displayed
    experiments_config = {
        'experiment_1': {
            'model_path': 'mlruns/mlruns_1/313770645939694569/7a56cdd2aecc410eac749071cdd68308/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 (Línea Base)'
        },
        'experiment_2': {
            'model_path': 'mlruns/mlruns_2/675918440420340658/48d40c1d5de64fd59dddb31484454246/artifacts/models/best_autoencoder_model.pth',
            'name': 'Perceptual'
        },
        'experiment_3': {
            'model_path': 'mlruns/mlruns_3/360562246159054027/725589bfcea14434a5ea4cb2a0c27dc3/artifacts/models/best_autoencoder_model.pth',
            'name': 'SSIM'
        },
        'experiment_4': {
            'model_path': 'mlruns/mlruns_4/702153582262418679/313e751960b44e48a74ca829ea6e66a1/artifacts/models/best_autoencoder_model.pth',
            'name': 'FFL'
        },
        'experiment_7': {
            'model_path': 'mlruns/mlruns_7/744827778112609234/7c8722fe646e465593e5028ff63ed29a/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + FFL'
        },
        'experiment_8': {
            'model_path': 'mlruns/mlruns_8/148501807127053673/c3d6cbcff4ed4cb39e5bf691407d12df/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + SSIM + FFL'
        },
        'experiment_10': {
            'model_path': 'mlruns/mlruns_10/819515919217557588/24198dcbfdbe49ccb4d554dd3751286f/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + Perceptual + SSIM + FFL'
        }
    }

    parser = argparse.ArgumentParser(description="Generate Volume 4x4 Grid comparison figure")
    parser.add_argument("--volume_index", type=int, default=22,
                       help="Index of the test volume to analyze (default: 22)")
    parser.add_argument("--output_dir", type=str, default="volume_grid_analysis",
                       help="Output directory for analysis results")

    args = parser.parse_args()

    try:
        analyzer = GridVolumeAnalyzer(
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
