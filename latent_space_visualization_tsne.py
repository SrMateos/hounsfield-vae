import os
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import torch.nn.functional as F

from monai.data import PersistentDataset, DataLoader
from monai.inferers import sliding_window_inference

# === IMPORTS ESPECÍFICOS DE TU PROYECTO (ajusta rutas si fuera necesario)
from src.networks.autoencoder_kl_sigmoid import AutoencoderKLSigmoid
from src.constants import ALL_TASKS
from src.data_handling import (
    get_data_paths,
    get_vae_val_transforms,
    make_transform_hash,
)
from src.data_handling.datasets import split_data


def setup_logging():
    """Configura el sistema de logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _get_default_config() -> Dict:
    """Devuelve una configuración por defecto para la inferencia."""
    return {
        "data": {
            "patch_size": [64, 64, 64],
            "train_split": 0.7,
            "val_split": 0.10,
            "test_split": 0.20,
            "task1": True
        },
        "inference": {
            "sw_batch_size": 20,
            "overlap": 0.75,
            "mode": "gaussian"
        },
        "debug": False
    }


def load_model_from_config(exp_config: Dict, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    """Carga un modelo y su configuración desde un diccionario."""
    model_path = exp_config['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en la ruta: {model_path}")

    model = AutoencoderKLSigmoid(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 64),
        latent_channels=3,
        num_res_blocks=1,
        norm_num_groups=16,
        attention_levels=(False, False, True),
    )

    logging.info(f"Cargando modelo desde: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Usar configuración por defecto ya que no la leemos de MLflow
    config = _get_default_config()

    return model, config


def prepare_test_loader(config: Dict):
    logging.info("Preparando dataset de test...")
    _, cts_paths, masks_paths = get_data_paths(ALL_TASKS, task1=config["data"]["task1"], debug=config["debug"])
    data = [{"image": ct, "mask": mask} for ct, mask in zip(cts_paths, masks_paths)]

    _, _, test_data = split_data(
        data=data,
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        random_seed=42,
    )

    val_transforms = get_vae_val_transforms(patch_size=tuple(config["data"]["patch_size"]))
    test_ds = PersistentDataset(data=test_data, transform=val_transforms, cache_dir="cache/test_tsne",
                                hash_transform=make_transform_hash(config["data"]["patch_size"]))
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True)
    return test_loader, val_transforms, test_data


def infer_latent_for_volume(
    model: torch.nn.Module,
    volume_tensor: torch.Tensor,
    device: torch.device,
    patch_size: Tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    mode: str,
    grid: int = 4,
) -> torch.Tensor:
    """
    Devuelve un vector 1D representando el latente del volumen:
    - Extrae z por sliding-window (latente tiene forma [B, C, H, W, D] por parche)
    - Agrega por TODOS los parches (media) tras hacer AdaptiveAvgPool3d(grid)
    - Resultado: vector de tamaño C * grid^3
    """
    def predictor_get_latent(x):
        with torch.no_grad():
            try:
                enc_out = model.encode(x)
            except AttributeError:
                raise RuntimeError("El modelo no expone el método .encode(x).")

            if isinstance(enc_out, (tuple, list)):
                z_mu = enc_out[0]
            elif isinstance(enc_out, dict) and ("mu" in enc_out):
                z_mu = enc_out["mu"]
            else:
                z_mu = enc_out
            return z_mu

    latent_maps = sliding_window_inference(
        inputs=volume_tensor.to(device),
        roi_size=tuple(patch_size),
        sw_batch_size=sw_batch_size,
        predictor=predictor_get_latent,
        overlap=overlap,
        mode=mode,
        device=device,
        sw_device=device,
    )

    pooled = F.adaptive_avg_pool3d(latent_maps, output_size=(grid, grid, grid))
    vec = pooled.flatten(start_dim=1)
    return vec.squeeze(0).detach().cpu()


def extract_anatomy_from_path(path: str) -> str:
    """Devuelve 'AB', 'HN' o 'TH' si aparece en el path; 'UNK' en otro caso."""
    if re.search(r'(?i)\bAB\b', path): return 'AB'
    if re.search(r'(?i)\bHN\b', path): return 'HN'
    if re.search(r'(?i)\bTH\b', path): return 'TH'
    lower = path.lower()
    if "ab" in lower: return "AB"
    if "hn" in lower: return "HN"
    if "th" in lower: return "TH"
    return 'UNK'


def balance_volumes_by_anatomy(test_data: List[Dict], max_volumes: int, seed: int = 42) -> List[Dict]:
    """Selecciona un subconjunto equilibrado de volúmenes por anatomía."""
    if max_volumes is None:
        return test_data

    np.random.seed(seed)
    anatomy_groups = defaultdict(list)
    for item in test_data:
        anatomy = extract_anatomy_from_path(str(item["image"]))
        anatomy_groups[anatomy].append(item)

    available_counts = {k: len(v) for k, v in anatomy_groups.items()}
    logging.info(f"Volúmenes disponibles por anatomía: {available_counts}")

    anatomies = list(anatomy_groups.keys())
    n_anatomies = len(anatomies)
    if n_anatomies == 0:
        return test_data[:max_volumes]

    base_per_anatomy = max_volumes // n_anatomies
    remainder = max_volumes % n_anatomies
    target_counts = {}
    for i, anatomy in enumerate(anatomies):
        target = base_per_anatomy + (1 if i < remainder else 0)
        target_counts[anatomy] = min(target, available_counts[anatomy])

    logging.info(f"Distribución objetivo: {target_counts}")

    selected_data = []
    for anatomy, count in target_counts.items():
        indices = np.random.choice(len(anatomy_groups[anatomy]), count, replace=False)
        selected_data.extend([anatomy_groups[anatomy][i] for i in indices])

    np.random.shuffle(selected_data)
    logging.info(f"Total de volúmenes seleccionados: {len(selected_data)}")
    return selected_data


def run_tsne_visualization(
    exp_config: Dict,
    exp_key: str,
    output_dir_base: str,
    grid: int,
    perplexity: float,
    early_exaggeration: float,
    learning_rate: str,
    n_iter: int,
    max_volumes: int,
    seed: int,
):
    output_dir = os.path.join(output_dir_base, exp_key)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_logging()

    model, cfg = load_model_from_config(exp_config, device)
    patch_size = cfg["data"]["patch_size"]
    sw_bs = cfg["inference"]["sw_batch_size"]
    overlap = cfg["inference"]["overlap"]
    mode = cfg["inference"]["mode"]

    test_loader, val_transforms, test_data = prepare_test_loader(cfg)

    if max_volumes is not None:
        balanced_test_data = balance_volumes_by_anatomy(test_data, max_volumes, seed)
        test_ds_balanced = PersistentDataset(data=balanced_test_data, transform=val_transforms, cache_dir=f"cache/test_tsne_{exp_key}",
                                               hash_transform=make_transform_hash(patch_size))
        test_loader = DataLoader(test_ds_balanced, batch_size=1, shuffle=False, pin_memory=True)

    features, labels, paths = [], [], []
    torch.manual_seed(seed)
    np.random.seed(seed)

    for i, batch in enumerate(test_loader):
        logging.info(f"Procesando volumen {i+1}/{len(test_loader)}...")
        vol = batch["image"]
        try:
            vec = infer_latent_for_volume(model, vol, device, patch_size, sw_bs, overlap, mode, grid)
            features.append(vec.numpy())
            img_path = batch["image_meta_dict"]["filename_or_obj"][0]
            paths.append(str(img_path))
            labels.append(extract_anatomy_from_path(str(img_path)))
        except RuntimeError as e:
            logging.error(f"No se pudo extraer el latente de un volumen: {e}")
            continue

    if not features:
        logging.error("No se extrajeron características latentes.")
        return

    X = np.vstack(features)
    y = np.array(labels)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lr_value = 'auto' if learning_rate == 'auto' else float(learning_rate)

    logging.info(f"Iniciando t-SNE con {len(X)} muestras...")
    reducer = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=lr_value, max_iter=n_iter, random_state=seed, verbose=1)
    emb = reducer.fit_transform(Xs)

    np.save(os.path.join(output_dir, "latent_features.npy"), X)

    import csv
    csv_path = os.path.join(output_dir, "latent_tsne.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "label", "path"])
        for (x1, x2), lab, pth in zip(emb, y, paths):
            w.writerow([float(x1), float(x2), lab, pth])

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    colors = {"AB": "tab:blue", "HN": "tab:orange", "TH": "tab:green", "UNK": "tab:gray"}
    for lab in np.unique(y):
        sel = y == lab
        ax.scatter(emb[sel, 0], emb[sel, 1], s=25, alpha=0.8, label=lab, c=colors.get(lab, "tab:gray"))
    ax.set_title(f"t-SNE del Latente - Experimento: {exp_config['name']}", fontsize=12)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.legend(title="Anatomía", loc="best")
    out_png = os.path.join(output_dir, "latent_tsne.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    info_path = os.path.join(output_dir, "info.txt")
    with open(info_path, "w") as f:
        f.write(f"Generated on: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Experiment: {exp_config['name']} ({exp_key})\n")
        f.write(f"Model Path: {exp_config['model_path']}\n")
        f.write(f"Samples: {len(X)}; Feature dim: {X.shape[1]}\n")
        f.write(f"Grid: {grid}; Patch size: {patch_size}; SW batch size: {sw_bs}; overlap: {overlap}; mode: {mode}\n")
        f.write(f"t-SNE: perplexity={perplexity}, early_exaggeration={early_exaggeration}, learning_rate={learning_rate}, n_iter={n_iter}\n")
        counts = Counter(y)
        f.write(f"Label counts: {dict(counts)}\n")
        if max_volumes is not None:
            f.write(f"Applied balanced sampling with max_volumes={max_volumes}\n")

    logging.info(f"Resultados guardados en: {output_dir}")


def main():
    # Define la configuración de todos los experimentos aquí
    # Las rutas asumen que el script se ejecuta desde el directorio raíz del proyecto
    EXPERIMENTS_CONFIG = {
        'exp_1': {
            'model_path': 'mlruns/mlruns_1/313770645939694569/7a56cdd2aecc410eac749071cdd68308/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 (Línea Base)'
        },
        'exp_2': {
            'model_path': 'mlruns/mlruns_2/675918440420340658/48d40c1d5de64fd59dddb31484454246/artifacts/models/best_autoencoder_model.pth',
            'name': 'Perceptual'
        },
        'exp_3': {
            'model_path': 'mlruns/mlruns_3/360562246159054027/725589bfcea14434a5ea4cb2a0c27dc3/artifacts/models/best_autoencoder_model.pth',
            'name': 'SSIM'
        },
        'exp_4': {
            'model_path': 'mlruns/mlruns_4/702153582262418679/313e751960b44e48a74ca829ea6e66a1/artifacts/models/best_autoencoder_model.pth',
            'name': 'FFL'
        },
        'exp_5': {
            'model_path': 'mlruns/mlruns_5/239927948431294758/ec5ae7551fff4204a529b7c7259fab65/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + Perceptual'
        },
        'exp_6': {
            'model_path': 'mlruns/mlruns_6/385784109692985319/f8de3076331c4e9fb5e3187ae54a7b2b/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + SSIM'
        },
        'exp_7': {
            'model_path': 'mlruns/mlruns_7/744827778112609234/7c8722fe646e465593e5028ff63ed29a/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + FFL'
        },
        'exp_8': {
            'model_path': 'mlruns/mlruns_8/148501807127053673/c3d6cbcff4ed4cb39e5bf691407d12df/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + SSIM + FFL'
        },
        'exp_9': {
            'model_path': 'mlruns/mlruns_9/346230370302623680/fda24d074c4c4d909160dc5e0c37c399/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + Perceptual + FFL'
        },
        'exp_10': {
            'model_path': 'mlruns/mlruns_10/819515919217557588/24198dcbfdbe49ccb4d554dd3751286f/artifacts/models/best_autoencoder_model.pth',
            'name': 'L1 + Perceptual + SSIM + FFL'
        }
    }

    parser = argparse.ArgumentParser(description="Visualización t-SNE del espacio latente de un modelo VAE.")
    parser.add_argument("--experiment_key", type=str, required=True, choices=EXPERIMENTS_CONFIG.keys(),
                        help="Clave del experimento a ejecutar (definido en el script).")
    parser.add_argument("--output_dir", type=str, default="latent_tsne_analysis", help="Directorio base para los resultados.")
    parser.add_argument("--grid", type=int, default=4, help="Tamaño del pooling adaptativo (reduce la dimensión latente a grid^3).")
    parser.add_argument("--perplexity", type=float, default=30.0, help="Perplejidad para t-SNE.")
    parser.add_argument("--early_exaggeration", type=float, default=12.0, help="Exageración temprana para t-SNE.")
    parser.add_argument("--learning_rate", type=str, default="auto", help="Tasa de aprendizaje para t-SNE ('auto' o un número).")
    parser.add_argument("--n_iter", type=int, default=1000, help="Número de iteraciones para t-SNE.")
    parser.add_argument("--max_volumes", type=int, default=252, help="Número máximo de volúmenes a procesar (aplica muestreo equilibrado).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")

    args = parser.parse_args()

    selected_experiment_config = EXPERIMENTS_CONFIG[args.experiment_key]

    try:
        run_tsne_visualization(
            exp_config=selected_experiment_config,
            exp_key=args.experiment_key,
            output_dir_base=args.output_dir,
            grid=args.grid,
            perplexity=args.perplexity,
            early_exaggeration=args.early_exaggeration,
            learning_rate=args.learning_rate,
            n_iter=args.n_iter,
            max_volumes=args.max_volumes,
            seed=args.seed,
        )
    except Exception as e:
        logging.error(f"Se ha producido un error durante la ejecución: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
