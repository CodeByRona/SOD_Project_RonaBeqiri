"""
train.py - Training and validation loop for Salient Object Detection.

Features:
  - Custom hybrid loss: Binary Cross-Entropy + 0.5 * (1 - IoU)
  - Per-epoch training AND validation loss logging
  - Early stopping when validation loss stops improving
  - Full checkpoint save/resume (weights + optimizer state + epoch)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from sod_model import SODModel
from tqdm import tqdm


# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

CHECKPOINT_PATH = "checkpoint.pth"
BEST_MODEL_PATH = "best_model.pth"


# ── Loss Function ─────────────────────────────────────────────────────────────
def custom_loss(pred, target):
    """
    Hybrid loss = BCE + 0.5 * (1 - IoU).
    BCE handles per-pixel accuracy; IoU handles structural overlap.
    """
    bce = nn.functional.binary_cross_entropy(pred, target)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-7)
    return bce + 0.5 * (1 - iou)


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_network():
    train_loader, val_loader, _ = load_data(batch_size=16)
    model = SODModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 30
    best_val_loss = float('inf')
    early_stop_patience = 5
    epochs_without_improvement = 0
    start_epoch = 0

    # ── Resume from checkpoint if it exists ───────────────────────────────────
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"[Checkpoint] Resumed from epoch {start_epoch}. Best val loss so far: {best_val_loss:.4f}")
    except FileNotFoundError:
        print("[Checkpoint] No checkpoint found. Starting training from scratch.")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc="  Training"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = custom_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="  Validation"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = custom_loss(outputs, masks)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ── Save checkpoint after every epoch (for resume) ────────────────────
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, CHECKPOINT_PATH)
        print(f"  [Checkpoint] Saved at epoch {epoch + 1}.")

        # ── Save best model based on validation loss ──────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  [Best Model] New best val loss: {best_val_loss:.4f} — best_model.pth saved.")
        else:
            epochs_without_improvement += 1
            print(f"  [Early Stop] No improvement for {epochs_without_improvement}/{early_stop_patience} epochs.")

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_without_improvement >= early_stop_patience:
            print(f"\n[Early Stop] Stopping training after {epoch + 1} epochs.")
            break

    print("\nTraining complete. Best model saved to:", BEST_MODEL_PATH)


if __name__ == "__main__":
    train_network()