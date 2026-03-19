"""
Обучение GRU-модели для генерации конфигураций ворот.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from gate_generator import generate_dataset, save_dataset, load_dataset
from gate_dataset import create_datasets
from gate_model import GateGRU, count_parameters

# Гиперпараметры
BATCH_SIZE = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15
N_SAMPLES = 10000

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_seq = batch["input_seq"].to(device)
        target_seq = batch["target_seq"].to(device)
        mask = batch["mask"].to(device)

        hidden = model.init_hidden(input_seq.size(0), device)
        pred, _ = model(input_seq, hidden)

        # MSE loss с маской (только на реальных элементах)
        loss_per_elem = (pred - target_seq) ** 2  # (batch, seq_len, 3)
        loss_per_step = loss_per_elem.mean(dim=-1)  # (batch, seq_len)
        masked_loss = (loss_per_step * mask).sum() / mask.sum()

        optimizer.zero_grad()
        masked_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += masked_loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_seq = batch["input_seq"].to(device)
        target_seq = batch["target_seq"].to(device)
        mask = batch["mask"].to(device)

        hidden = model.init_hidden(input_seq.size(0), device)
        pred, _ = model(input_seq, hidden)

        loss_per_elem = (pred - target_seq) ** 2
        loss_per_step = loss_per_elem.mean(dim=-1)
        masked_loss = (loss_per_step * mask).sum() / mask.sum()

        total_loss += masked_loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Данные
    data_path = DATA_DIR / "gate_configs.npz"
    if data_path.exists():
        print("Загрузка датасета...")
        configs = load_dataset(data_path)
    else:
        print("Генерация датасета...")
        configs = generate_dataset(n_samples=N_SAMPLES)
        save_dataset(configs, data_path)

    train_ds, val_ds = create_datasets(configs)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Модель
    model = GateGRU(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    print(f"Параметров: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ClearML (опционально)
    try:
        from clearml import Task
        task = Task.init(project_name="DroneTrack", task_name="GRU Gate Placement v1")
        task.connect({
            "batch_size": BATCH_SIZE, "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS, "dropout": DROPOUT,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS, "n_samples": N_SAMPLES,
        })
        logger = task.get_logger()
        use_clearml = True
    except ImportError:
        use_clearml = False
        print("ClearML не установлен, логирование отключено")

    # Обучение
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = val_epoch(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr_now:.6f}")

        if use_clearml:
            logger.report_scalar("Loss", "train", train_loss, epoch)
            logger.report_scalar("Loss", "val", val_loss, epoch)
            logger.report_scalar("LR", "lr", lr_now, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print(f"  -> Лучшая модель сохранена (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping на эпохе {epoch}")
                break

    print(f"\nОбучение завершено. Лучший val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
