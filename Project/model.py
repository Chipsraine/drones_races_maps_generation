"""
Архитектура модели: U-Net для генерации траекторий на карте трассы.

Задача: сегментация — для каждой клетки матрицы предсказать,
является ли она частью пути дрона.

Вход:  (B, 5, H, W) — one-hot encoded карта (5 классов: пусто/путь/ворота/кольцо/столбик)
Выход: (B, 1, H, W) — вероятность того, что клетка является путём
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Строительные блоки ───────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Двойная свёртка с BatchNorm и ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool → ConvBlock (уменьшение пространственного размера в 2 раза)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Bilinear upsample → concat со skip-connection → ConvBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Выравниваем размеры если нужно (на случай нечётных размеров)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─── U-Net ────────────────────────────────────────────────────────────────────

class TrackUNet(nn.Module):
    """
    U-Net для предсказания пути на карте трассы.

    Параметры:
        n_classes_in  — количество классов элементов (5: пусто/путь/ворота/кольцо/столбик)
        base_features — базовое количество фильтров (по умолчанию 32)
    """

    def __init__(self, n_classes_in=5, base_features=32):
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = ConvBlock(n_classes_in, f)      # 100×100 → 100×100
        self.enc2 = Down(f,     f * 2)              # → 50×50
        self.enc3 = Down(f * 2, f * 4)              # → 25×25
        self.enc4 = Down(f * 4, f * 8)              # → 12×12

        # Bottleneck
        self.bottleneck = Down(f * 8, f * 16)       # → 6×6

        # Decoder
        self.dec4 = Up(f * 16 + f * 8, f * 8)      # → 12×12
        self.dec3 = Up(f * 8  + f * 4, f * 4)      # → 25×25
        self.dec2 = Up(f * 4  + f * 2, f * 2)      # → 50×50
        self.dec1 = Up(f * 2  + f,     f)           # → 100×100

        # Выходной слой: предсказываем вероятность пути для каждой клетки
        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder с skip-connections
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)   # (B, 1, H, W) — logits


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def grid_to_tensor(grid: "np.ndarray") -> "torch.Tensor":
    """
    Конвертирует матрицу (H, W) с кодами 0-4 в one-hot тензор (5, H, W).
    """
    import numpy as np
    n_classes = 5
    one_hot = np.zeros((n_classes, *grid.shape), dtype=np.float32)
    for cls in range(n_classes):
        one_hot[cls] = (grid == cls).astype(np.float32)
    return torch.from_numpy(one_hot)


def get_model_info(model: nn.Module) -> dict:
    """Возвращает информацию о модели для логирования в ClearML."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "architecture": "U-Net",
    }


# ─── Быстрая проверка ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    model = TrackUNet(n_classes_in=5, base_features=32)
    info = get_model_info(model)
    print(f"Архитектура: {info['architecture']}")
    print(f"Параметров всего: {info['total_parameters']:,}")
    print(f"Обучаемых: {info['trainable_parameters']:,}")

    # Тестовый прогон
    dummy_input = torch.zeros(2, 5, 100, 100)   # batch=2, 5 классов, 100×100
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nВход:  {tuple(dummy_input.shape)}")
    print(f"Выход: {tuple(output.shape)}")
    print("Модель работает корректно.")
