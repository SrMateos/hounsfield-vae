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
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

from monai.data import PersistentDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms.utils import allow_missing_keys_mode
from monai.data import decollate_batch
import torch.nn.functional as F

# === IMPORTS ESPECÍFICOS DE TU PROYECTO (ajusta rutas si fuera necesario)
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


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def find_best_run(experiment_id: str) -> str:
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.best_val_recon_loss ASC"],
        max_results=1
    )
    if not runs:
        raise ValueError(f"No runs found in experiment {experiment_id}")
    best = runs[0].info.run_id
    logging.info(f"Best run for experiment {experiment_id}: {best}")
    return best


def get_experiment_name(experiment_id: str) -> str:
    try:
        client = mlflow.tracking.MlflowClient()
        return client.get_experiment(experiment_id).name
    except Exception:
        return f"experiment_{experiment_id}"


def load_config_from_run(run_id: str) -> Dict:
    return load_run_config(run_id)


def load_model(experiment_id: str, run_id: str, mlruns_path: str, device: torch.device):
    if run_id is None:
        run_id = find_best_run(experiment_id)

    config = load_config_from_run(run_id)

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

    model_path = os.path.join(mlruns_path, experiment_id, run_id, "artifacts", "models", "best_autoencoder_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")

    logging.info(f"Loading model from: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config, run_id


def prepare_test_loader(config: Dict):
    logging.info("Preparing test dataset...")
    _, cts_paths, masks_paths = get_data_paths(ALL_TASKS, task1=config["data"]["task1"], debug=config["debug"])
    data = [{"image": ct, "mask": mask} for ct, mask in zip(cts_paths, masks_paths)]

    train_data, val_data, test_data = split_data(
        data=data,
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        random_seed=42,
    )

    val_transforms = get_vae_val_transforms(patch_size=tuple(config["data"]["patch_size"]))
    test_ds = PersistentDataset(data=test_data, transform=val_transforms, cache_dir="cache/test_umap",
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
    max_patches: int = None,
) -> torch.Tensor:
    """
    Devuelve un vector 1D representando el latente del volumen:
    - Extrae z por sliding-window (latente tiene forma [B, C=3, 16, 16, 16] por parche)
    - Agrega por TODOS los parches (media) tras hacer AdaptiveAvgPool3d(grid)
    - Resultado: vector de tamaño C * grid^3
    """

    # Función "predictor" que devuelve el latente en vez de la reconstrucción.
    # Intentamos usar .encode() y quedarnos con la media (mu). Ajusta si tu clase retorna otra cosa.
    def predictor_get_latent(x):
        with torch.no_grad():
            # Caso típico VAE: encode -> (mu, logvar) o dict
            try:
                enc_out = model.encode(x)  # [B, C, 16, 16, 16] o tu estructura
            except AttributeError:
                # Si tu modelo no expone .encode, puedes crear un método 'reconstruct' que devuelva también latente
                raise RuntimeError("El modelo no expone .encode(x). Implementa/expone encode o añade un hook.")

            # Normalizamos varias posibles salidas:
            if isinstance(enc_out, (tuple, list)) and len(enc_out) >= 1:
                z_mu = enc_out[0]
            elif isinstance(enc_out, dict) and ("mu" in enc_out):
                z_mu = enc_out["mu"]
            else:
                # Si encode ya devuelve el latente determinístico
                z_mu = enc_out

            # z_mu: [B, 3, 16,16,16]
            return z_mu

    # Hacemos sliding-window sobre el VOLUMEN para sacar latentes por parche
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
    # latent_maps: [1, 3, 16,16,16] "compuesto" por agregación espacial de parches según el modo.
    # PERO: sliding_window_inference al usar 'mode' combina salidas por voxel; aquí estamos combinando latentes
    # que viven en un espacio de 16^3 por parche. Para evitar mezclar latentes de distinta posición,
    # una estrategia práctica es: tomar la media global de z por "parche" antes de que el SWI combine.
    # Como SWI no nos da cada parche por separado, usamos el resultado combinado como representación global.
    # Si quisieras muestrear parches explícitos, habría que iterar manualmente.

    # Pool adaptativo a grid^3 para reducir 16^3 -> grid^3 por canal
    pooled = F.adaptive_avg_pool3d(latent_maps, output_size=(grid, grid, grid))  # [1, 3, g, g, g]
    vec = pooled.flatten(start_dim=1)  # [1, 3*g*g*g]
    return vec.squeeze(0).detach().cpu()  # [F]


def extract_anatomy_from_path(path: str) -> str:
    """Devuelve 'AB', 'HN' o 'TH' si aparece en el path; 'UNK' en otro caso."""
    if re.search(r'(?i)\bAB\b', path):
        return 'AB'
    if re.search(r'(?i)\bHN\b', path):
        return 'HN'
    if re.search(r'(?i)\bTH\b', path):
        return 'TH'
    # fallback: busca como substring (menos estricto)
    lower = path.lower()
    if "ab" in lower:
        return "AB"
    if "hn" in lower:
        return "HN"
    if "th" in lower:
        return "TH"
    return 'UNK'


def balance_volumes_by_anatomy(test_data: List[Dict], max_volumes: int, seed: int = 42) -> List[Dict]:
    """
    Selecciona un subconjunto equilibrado de volúmenes por anatomía.

    Args:
        test_data: Lista de diccionarios con 'image' y 'mask'
        max_volumes: Número máximo total de volúmenes
        seed: Semilla para reproducibilidad

    Returns:
        Lista equilibrada de datos
    """
    if max_volumes is None:
        return test_data

    np.random.seed(seed)

    # Agrupa los datos por anatomía
    anatomy_groups = defaultdict(list)
    for item in test_data:
        img_path = str(item["image"])
        anatomy = extract_anatomy_from_path(img_path)
        anatomy_groups[anatomy].append(item)

    # Cuenta disponible por anatomía
    available_counts = {anatomy: len(items) for anatomy, items in anatomy_groups.items()}
    logging.info(f"Volúmenes disponibles por anatomía: {available_counts}")

    # Calcula distribución objetivo
    anatomies = list(anatomy_groups.keys())
    n_anatomies = len(anatomies)

    if n_anatomies == 0:
        logging.warning("No se encontraron anatomías identificables")
        return test_data[:max_volumes]

    # Distribución base equitativa
    base_per_anatomy = max_volumes // n_anatomies
    remainder = max_volumes % n_anatomies

    # Asigna volúmenes por anatomía
    target_counts = {}
    for i, anatomy in enumerate(anatomies):
        target = base_per_anatomy
        if i < remainder:  # Distribuye el resto
            target += 1
        # No puede exceder lo disponible
        target_counts[anatomy] = min(target, available_counts[anatomy])

    logging.info(f"Distribución objetivo: {target_counts}")

    # Selecciona aleatoriamente de cada grupo
    selected_data = []
    actual_counts = {}

    for anatomy, target_count in target_counts.items():
        if target_count > 0:
            available_items = anatomy_groups[anatomy]
            if len(available_items) > target_count:
                # Selección aleatoria sin reemplazo
                selected_items = np.random.choice(
                    available_items,
                    size=target_count,
                    replace=False
                ).tolist()
            else:
                selected_items = available_items

            selected_data.extend(selected_items)
            actual_counts[anatomy] = len(selected_items)

    # Mezcla el orden final
    np.random.shuffle(selected_data)

    logging.info(f"Distribución final: {actual_counts}")
    logging.info(f"Total de volúmenes seleccionados: {len(selected_data)}")

    return selected_data


def run_umap_visualization(
    experiment_id: str,
    run_id: str,
    mlruns_path: str,
    output_dir: str,
    grid: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    max_volumes: int,
    seed: int,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_logging()

    # 1) Carga modelo y config
    model, cfg, run_id = load_model(experiment_id, run_id, mlruns_path, device)
    patch_size = cfg["data"]["patch_size"]
    sw_bs = cfg["inference"]["sw_batch_size"]
    overlap = cfg["inference"]["overlap"]
    mode = cfg["inference"]["mode"]

    # 2) Data
    test_loader, val_transforms, test_data = prepare_test_loader(cfg)

    # 3) NUEVO: Equilibra los volúmenes por anatomía si se especifica max_volumes
    if max_volumes is not None:
        balanced_test_data = balance_volumes_by_anatomy(test_data, max_volumes, seed)
        # Crea nuevo dataset con los datos equilibrados
        test_ds_balanced = PersistentDataset(
            data=balanced_test_data,
            transform=val_transforms,
            cache_dir="cache/test_umap_balanced",
            hash_transform=make_transform_hash(patch_size),
        )
        test_loader = DataLoader(test_ds_balanced, batch_size=1, shuffle=False, pin_memory=True)

    features = []
    labels = []
    paths = []

    # 4) Procesa todos los volúmenes (ya equilibrados si era necesario)
    torch.manual_seed(seed)
    np.random.seed(seed)

    for batch in test_loader:
        # batch: dict con 'image' y 'mask'; mantenemos preprocesado
        vol = batch["image"]  # [1, 1, Dx, Hx, Wx] (MONAI ordena como [N, C, Z, Y, X])
        try:
            vec = infer_latent_for_volume(
                model=model,
                volume_tensor=vol,
                device=device,
                patch_size=patch_size,
                sw_batch_size=sw_bs,
                overlap=overlap,
                mode=mode,
                grid=grid,
            )
        except RuntimeError as e:
            logging.error(f"No se pudo extraer el latente de un volumen: {e}")
            continue

        features.append(vec.numpy())
        # detecta anatomía desde el path original
        img_path = batch["image_meta_dict"]["filename_or_obj"][0] if "image_meta_dict" in batch else ""
        paths.append(str(img_path))
        labels.append(extract_anatomy_from_path(str(img_path)))

    if len(features) == 0:
        logging.error("No se extrajeron características latentes. Revisa el flujo de datos.")
        return

    X = np.vstack(features)  # [N, F]
    y = np.array(labels)

    # 5) Escalado + UMAP
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        n_components=2,
        verbose=True
    )
    emb = reducer.fit_transform(Xs)  # [N, 2]

    # 6) Guardados
    np.save(os.path.join(output_dir, "latent_features.npy"), X)
    # CSV con embedding y etiqueta
    import csv
    csv_path = os.path.join(output_dir, "latent_umap.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "label", "path"])
        for (x1, x2), lab, pth in zip(emb, y, paths):
            w.writerow([float(x1), float(x2), lab, pth])

    # 7) Plot
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    colors = {"AB": "tab:blue", "HN": "tab:orange", "TH": "tab:green", "UNK": "tab:gray"}
    for lab in np.unique(y):
        sel = y == lab
        ax.scatter(emb[sel, 0], emb[sel, 1], s=25, alpha=0.8, label=lab, c=colors.get(lab, "tab:gray"))
    ax.set_title(f"UMAP del latente (grid={grid}³) – exp {experiment_id} / run {run_id}", fontsize=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(title="Anatomía", loc="best")
    out_png = os.path.join(output_dir, "latent_umap.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # 8) Log
    info_path = os.path.join(output_dir, "info.txt")
    with open(info_path, "w") as f:
        f.write(f"Generated on: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Experiment: {get_experiment_name(experiment_id)} ({experiment_id})\nRun: {run_id}\n")
        f.write(f"Samples: {len(X)}; Feature dim: {X.shape[1]}\n")
        f.write(f"Grid: {grid}; Patch size: {patch_size}; SW batch size: {sw_bs}; overlap: {overlap}; mode: {mode}\n")
        f.write(f"UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}\n")
        counts = {lab: int(np.sum(y == lab)) for lab in np.unique(y)}
        f.write(f"Label counts: {counts}\n")
        if max_volumes is not None:
            f.write(f"Applied balanced sampling with max_volumes={max_volumes}\n")

    logging.info(f"Saved:\n - {out_png}\n - {csv_path}\n - {os.path.join(output_dir, 'latent_features.npy')}\n - {info_path}")


def main():
    parser = argparse.ArgumentParser(description="UMAP del estado latente por anatomías (AB/HN/TH)")
    parser.add_argument("--experiment_id", type=str, required=True, help="ID del experimento en MLflow")
    parser.add_argument("--run_id", type=str, default=None, help="ID del run concreto (opcional)")
    parser.add_argument("--mlruns_path", type=str, default="mlruns", help="Ruta al directorio mlruns")
    parser.add_argument("--output_dir", type=str, default="latent_umap_analysis", help="Directorio de salida")

    parser.add_argument("--grid", type=int, default=4, help="Tamaño de pooling adaptativo por eje (reduce 16→grid)")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--metric", type=str, default="euclidean", help="UMAP metric")

    parser.add_argument("--max_volumes", type=int, default=None, help="Máximo de volúmenes a procesar (con balance)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")

    args = parser.parse_args()

    try:
        run_umap_visualization(
            experiment_id=args.experiment_id,
            run_id=args.run_id,
            mlruns_path=args.mlruns_path,
            output_dir=args.output_dir,
            grid=args.grid,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            max_volumes=args.max_volumes,
            seed=args.seed,
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
