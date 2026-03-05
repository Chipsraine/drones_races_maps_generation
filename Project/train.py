"""
Обучение U-Net для предсказания тепловой карты ключевых точек (waypoints).
Все метрики, параметры и модель логируются в ClearML.

Запуск:
    uv run python Project/train.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from clearml import Task, Dataset as ClearMLDataset

from model import TrackUNet, grid_to_tensor, get_model_info
from dataset_generator import generate_dataset


# ─── Конфигурация ─────────────────────────────────────────────────────────────

HYPERPARAMS = {
    # Данные
    "n_samples":    2000,
    "grid_size":    100,
    "n_gates":      2,
    "n_rings":      2,
    "n_poles":      3,
    "train_ratio":  0.8,
    "seed":         42,
    # Модель
    "base_features": 32,
    # Обучение
    "epochs":           50,
    "batch_size":       8,
    "lr":               1e-3,
    "weight_decay":     1e-4,
    # Early stopping
    "early_stop_patience": 10,
}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TrackDataset(Dataset):
    """
    Датасет пар (input, heatmap).
    input:   карта без пути → one-hot тензор (5, H, W)
    heatmap: тепловая карта waypoints → тензор (1, H, W) float32, значения [0, 1]
    """

    def __init__(self, inputs: np.ndarray, heatmaps: np.ndarray):
        self.inputs   = inputs
        self.heatmaps = heatmaps

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = grid_to_tensor(self.inputs[idx])                          # (5, H, W)
        y = torch.from_numpy(self.heatmaps[idx]).unsqueeze(0)        # (1, H, W)
        return x, y


# ─── Обучение ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_peak_sim = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()

            # Peak similarity: корреляция между предсказанной и целевой тепловой картой
            pred_sig = torch.sigmoid(pred)
            # Нормализуем оба до [0,1] по батчу
            p_flat = pred_sig.view(pred_sig.size(0), -1)
            y_flat = y.view(y.size(0), -1)
            # Cosine similarity по пространственным картам
            cos_sim = torch.nn.functional.cosine_similarity(p_flat, y_flat, dim=1).mean()
            total_peak_sim += cos_sim.item()

    return total_loss / len(loader), total_peak_sim / len(loader)


# ─── Главная функция ──────────────────────────────────────────────────────────

def main():
    # ── Инициализация ClearML задачи ──
    task = Task.init(
        project_name="DroneTrack",
        task_name="UNet Heatmap Waypoints v3",
    )
    task.add_tags(["unet", "heatmap", "waypoints", "bfs"])
    task.connect(HYPERPARAMS)
    logger = task.get_logger()

    hp = HYPERPARAMS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # ── Данные ──
    os.makedirs("data", exist_ok=True)
    if (os.path.exists("data/inputs.npy")
            and os.path.exists("data/heatmaps.npy")
            and os.path.exists("data/paths.npy")):
        print("Загрузка существующего датасета...")
        inputs   = np.load("data/inputs.npy")
        heatmaps = np.load("data/heatmaps.npy")
        print(f"Загружено {len(inputs)} примеров из data/")
    else:
        print("Генерация датасета...")
        inputs, heatmaps, paths = generate_dataset(
            n_samples=hp["n_samples"],
            grid_size=hp["grid_size"],
            n_gates=hp["n_gates"],
            n_rings=hp["n_rings"],
            n_poles=hp["n_poles"],
            seed=hp["seed"],
        )
        np.save("data/inputs.npy",   inputs)
        np.save("data/heatmaps.npy", heatmaps)
        np.save("data/paths.npy",    paths)

        clearml_dataset = ClearMLDataset.create(
            dataset_project="DroneTrack",
            dataset_name="Synthetic Tracks v3",
        )
        clearml_dataset.add_files("data/inputs.npy")
        clearml_dataset.add_files("data/heatmaps.npy")
        clearml_dataset.add_files("data/paths.npy")
        clearml_dataset.upload()
        clearml_dataset.finalize()
        print(f"Датасет загружен в ClearML (ID: {clearml_dataset.id})")

    # ── DataLoader ──
    dataset = TrackDataset(inputs, heatmaps)
    n_train = int(len(dataset) * hp["train_ratio"])
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(hp["seed"])
    )
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False, num_workers=0)
    print(f"Train: {n_train} | Val: {n_val}")

    # ── Модель ──
    model = TrackUNet(n_classes_in=5, base_features=hp["base_features"]).to(device)
    info  = get_model_info(model)
    print(f"Параметров модели: {info['total_parameters']:,}")
    logger.report_table(
        title="Model Info", series="Architecture",
        iteration=0,
        table_plot=pd.DataFrame({"Property": list(info.keys()), "Value": list(info.values())}),
    )

    # MSE loss для регрессии тепловой карты
    # Sigmoid применяется внутри — используем BCEWithLogitsLoss (лучше числено)
    # Target: нормализованная тепловая карта [0, 1]
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])

    # ── Цикл обучения ──
    best_val_sim    = 0.0
    patience_counter = 0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, hp["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_sim = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.report_scalar("Loss",           "train", value=train_loss, iteration=epoch)
        logger.report_scalar("Loss",           "val",   value=val_loss,   iteration=epoch)
        logger.report_scalar("Peak Similarity","val",   value=val_sim,    iteration=epoch)
        logger.report_scalar("LR",             "lr",    value=scheduler.get_last_lr()[0], iteration=epoch)

        improved = val_sim > best_val_sim
        marker   = " ★" if improved else ""
        print(f"Epoch {epoch:02d}/{hp['epochs']} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_sim={val_sim:.4f}{marker}")

        if improved:
            best_val_sim     = val_sim
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= hp["early_stop_patience"]:
            print(f"\nEarly stopping: similarity не улучшалась {hp['early_stop_patience']} эпох подряд.")
            break

    task.upload_artifact("best_model", "models/best_model.pt")
    logger.report_single_value("best_val_similarity", best_val_sim)
    print(f"\nОбучение завершено! Best val similarity: {best_val_sim:.4f}")
    print("Метрики доступны в ClearML: https://app.clear.ml")

    task.close()


if __name__ == "__main__":
    main()
