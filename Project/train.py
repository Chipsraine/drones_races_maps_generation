"""
Обучение U-Net для генерации траекторий на карте трассы.
Все метрики, параметры и модель логируются в ClearML.

Запуск:
    uv run python Project/train.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from clearml import Task, Dataset as ClearMLDataset

from model import TrackUNet, grid_to_tensor, get_model_info
from dataset_generator import generate_dataset


# ─── Конфигурация ─────────────────────────────────────────────────────────────

HYPERPARAMS = {
    # Данные
    "n_samples":    500,
    "grid_size":    100,
    "n_gates":      2,
    "n_rings":      2,
    "n_poles":      3,
    "train_ratio":  0.8,
    "seed":         42,
    # Модель
    "base_features": 32,
    # Обучение
    "epochs":        20,
    "batch_size":    8,
    "lr":            1e-3,
    "weight_decay":  1e-4,
}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TrackDataset(Dataset):
    """
    Датасет пар (input, target).
    input:  карта без пути → one-hot тензор (5, H, W)
    target: маска пути     → бинарный тензор (1, H, W)
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs  = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = grid_to_tensor(self.inputs[idx])           # (5, H, W)
        y = (self.targets[idx] == 1).astype(np.float32)  # бинарная маска пути
        y = torch.from_numpy(y).unsqueeze(0)           # (1, H, W)
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
    total_iou  = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()

            # IoU для пути (класс 1)
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
            intersection = (pred_bin * y).sum()
            union = (pred_bin + y).clamp(0, 1).sum()
            iou = (intersection / (union + 1e-6)).item()
            total_iou += iou

    return total_loss / len(loader), total_iou / len(loader)


# ─── Главная функция ──────────────────────────────────────────────────────────

def main():
    # ── Инициализация ClearML задачи ──
    task = Task.init(
        project_name="DroneTrack",
        task_name="UNet Path Generator v1",
        output_uri=True,
    )
    task.add_tags(["unet", "path-generation", "preliminary"])
    task.connect(HYPERPARAMS)
    logger = task.get_logger()

    hp = HYPERPARAMS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # ── Данные ──
    print("Генерация датасета...")
    inputs, targets = generate_dataset(
        n_samples=hp["n_samples"],
        grid_size=hp["grid_size"],
        n_gates=hp["n_gates"],
        n_rings=hp["n_rings"],
        n_poles=hp["n_poles"],
        seed=hp["seed"],
    )

    # Сохраняем локально и загружаем в ClearML Dataset
    os.makedirs("data", exist_ok=True)
    np.save("data/inputs.npy", inputs)
    np.save("data/targets.npy", targets)

    clearml_dataset = ClearMLDataset.create(
        dataset_project="DroneTrack",
        dataset_name="Synthetic Tracks",
    )
    clearml_dataset.add_files("data/inputs.npy")
    clearml_dataset.add_files("data/targets.npy")
    clearml_dataset.upload()
    clearml_dataset.finalize()
    print(f"Датасет загружен в ClearML (ID: {clearml_dataset.id})")

    # ── DataLoader ──
    dataset = TrackDataset(inputs, targets)
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
        table_plot={"Property": list(info.keys()), "Value": list(info.values())},
    )

    # Используем BCEWithLogitsLoss + pos_weight для несбалансированных данных
    # (путь занимает малую часть карты)
    pos_weight = torch.tensor([10.0]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])

    # ── Цикл обучения ──
    best_val_iou = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, hp["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # Логируем в ClearML
        logger.report_scalar("Loss", "train", value=train_loss, iteration=epoch)
        logger.report_scalar("Loss", "val",   value=val_loss,   iteration=epoch)
        logger.report_scalar("IoU (path)", "val", value=val_iou, iteration=epoch)
        logger.report_scalar("LR", "lr",
                             value=scheduler.get_last_lr()[0], iteration=epoch)

        print(f"Epoch {epoch:02d}/{hp['epochs']} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_iou={val_iou:.4f}")

        # Сохраняем лучшую модель
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), "models/best_model.pt")

    logger.report_single_value("best_val_iou", best_val_iou)
    print(f"\nОбучение завершено! Best val IoU: {best_val_iou:.4f}")
    print("Метрики доступны в ClearML: https://app.clear.ml")

    task.close()


if __name__ == "__main__":
    main()
