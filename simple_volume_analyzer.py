import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
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


class Simple2x2VolumeAnalyzer:
    """Simple analyzer that creates a 2x2 grid with original and 3 specific predictions"""

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

    def _generate_2x2_figure(self, output_dir):
        """Generate simple 2x2 grid figure with original and 3 predictions"""
        logging.info("Generating 2x2 grid figure...")

        exp_keys = list(self.experiment_data.keys())
        if len(exp_keys) == 0:
            raise ValueError("No experiments processed successfully")

        # Get reference original (should be same for all)
        reference_original = list(self.experiment_data.values())[0]['original']

        # Get predictions in order
        predictions = []
        exp_names = []
        for exp_key in exp_keys:
            exp_data = self.experiment_data[exp_key]
            predictions.append(exp_data['predicted'])
            exp_names.append(exp_data['name'])

        # Calculate global intensity range for consistency
        all_images = [reference_original] + predictions
        vmin_global, vmax_global = np.percentile(np.concatenate([img.flatten() for img in all_images]), [1, 99])

        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)

        # Position (0,0): Original image
        axes[0, 0].imshow(reference_original, cmap='gray', vmin=vmin_global, vmax=vmax_global)
        axes[0, 0].set_title('Volumen Original', fontsize=16, fontweight='bold')
        axes[0, 0].axis('off')

        # Positions (0,1), (1,0), (1,1): Predictions from experiments 5, 6, 9
        positions = [(0, 1), (1, 0), (1, 1)]

        for i, (prediction, exp_name) in enumerate(zip(predictions, exp_names)):
            if i < 3:  # Only show first 3 experiments
                row, col = positions[i]
                axes[row, col].imshow(prediction, cmap='gray', vmin=vmin_global, vmax=vmax_global)
                axes[row, col].set_title(f'{exp_name}', fontsize=14, fontweight='bold')
                axes[row, col].axis('off')

        # Add a single colorbar for intensity
        im = axes[0, 0].get_images()[0]
        cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=20,
                           label='Intensidad', location='right', pad=0.02)

        # # Add main title
        # fig.suptitle(f'Volumen {self.volume_index}: Original vs Reconstrucciones Seleccionadas',
        #              fontsize=18, fontweight='bold', y=0.95)

        # Save figure
        output_path = os.path.join(output_dir, f"volume_{self.volume_index}_2x2_comparison.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        logging.info(f"2x2 figure saved to: {output_path}")
        return output_path

    def run(self, output_dir="volume_2x2_analysis"):
        """Main execution method"""
        logging.info(f"Starting Volume {self.volume_index} 2x2 analysis...")
        logging.info(f"Device: {self.device}")

        os.makedirs(output_dir, exist_ok=True)

        # Process each experiment
        for exp_key, exp_config in self.experiments_config.items():
            try:
                self._process_experiment(exp_key, exp_config)
            except Exception as e:
                logging.error(f"Failed to process experiment {exp_key}: {str(e)}")
                continue

        # Generate 2x2 figure
        if len(self.experiment_data) >= 1:
            figure_path = self._generate_2x2_figure(output_dir)
            logging.info("2x2 analysis completed successfully!")
            return figure_path
        else:
            logging.error("No experiments processed successfully!")
            return None


def main():
    # Define experiments 5, 6, and 9 configuration
    experiments_config = {
        'experiment_5': {
            'model_path': 'mlruns/mlruns_5/239927948431294758/ec5ae7551fff4204a529b7c7259fab65/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + Perceptual (Caso 5)'
        },
        'experiment_6': {
            'model_path': 'mlruns/mlruns_6/385784109692985319/f8de3076331c4e9fb5e3187ae54a7b2b/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + SSIM (Caso 6)'
        },
        'experiment_9': {
            'model_path': 'mlruns/mlruns_9/346230370302623680/fda24d074c4c4d909160dc5e0c37c399/artifacts/models/best_autoencoder_model.pth',
            'name': 'Perceptual + SSIM + FFL (Caso 9)'
        }
    }

    parser = argparse.ArgumentParser(description="Generate Volume 2x2 comparison figure")
    parser.add_argument("--volume_index", type=int, default=31,
                       help="Index of the test volume to analyze (default: 22)")
    parser.add_argument("--output_dir", type=str, default="volume_2x2_analysis",
                       help="Output directory for analysis results")

    args = parser.parse_args()

    try:
        analyzer = Simple2x2VolumeAnalyzer(
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
