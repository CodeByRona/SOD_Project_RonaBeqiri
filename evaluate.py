"""
evaluate.py - Evaluation metrics and visualization for Salient Object Detection.

Computes on the test set:
  - Mean IoU (Intersection over Union)
  - Mean Precision
  - Mean Recall
  - Mean F1-Score

Also saves a visualization grid showing:
  - Input image
  - Ground-truth mask
  - Predicted saliency map
  - Binary predicted mask
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data
from sod_model import SODModel


def run_evaluation():
    """Evaluate the best saved model on the test set and visualize sample predictions."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────────
    _, _, test_loader = load_data(batch_size=1)
    model = SODModel().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # ── Metric accumulators ───────────────────────────────────────────────────
    iou_scores       = []
    precision_scores = []
    recall_scores    = []
    f1_scores        = []

    # For visualization: store first 4 samples
    viz_samples = []

    print("Evaluating on test set...")

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = (outputs > 0.5).float()

            # ── IoU ───────────────────────────────────────────────────────────
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum() - intersection
            iou = (intersection + 1e-7) / (union + 1e-7)
            iou_scores.append(iou.item())

            # ── Precision, Recall, F1 ─────────────────────────────────────────
            tp = (preds * masks).sum()
            fp = (preds * (1 - masks)).sum()
            fn = ((1 - preds) * masks).sum()

            precision = tp / (tp + fp + 1e-7)
            recall    = tp / (tp + fn + 1e-7)
            f1        = 2 * (precision * recall) / (precision + recall + 1e-7)

            precision_scores.append(precision.item())
            recall_scores.append(recall.item())
            f1_scores.append(f1.item())

            # ── Collect samples for visualization ─────────────────────────────
            if len(viz_samples) < 4:
                viz_samples.append({
                    'image':  imgs.squeeze().cpu().permute(1, 2, 0).numpy(),
                    'gt':     masks.squeeze().cpu().numpy(),
                    'pred':   outputs.squeeze().cpu().numpy(),
                    'binary': preds.squeeze().cpu().numpy(),
                })

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n========== Final Evaluation Results ==========")
    print(f"  Mean IoU       : {np.mean(iou_scores):.4f}")
    print(f"  Mean Precision : {np.mean(precision_scores):.4f}")
    print(f"  Mean Recall    : {np.mean(recall_scores):.4f}")
    print(f"  Mean F1-Score  : {np.mean(f1_scores):.4f}")
    print("==============================================")

    # ── Visualization grid ────────────────────────────────────────────────────
    n = len(viz_samples)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    fig.suptitle("Test Set Predictions: Input | Ground Truth | Predicted Saliency | Binary Mask", fontsize=12)

    col_titles = ["Input Image", "Ground-Truth Mask", "Predicted Saliency", "Binary Mask (0.5)"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=10, fontweight='bold')

    for row, sample in enumerate(viz_samples):
        axes[row][0].imshow(np.clip(sample['image'], 0, 1))
        axes[row][1].imshow(sample['gt'],     cmap='gray')
        axes[row][2].imshow(sample['pred'],   cmap='jet')
        axes[row][3].imshow(sample['binary'], cmap='gray')

        for col in range(4):
            axes[row][col].axis('off')

    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved to evaluation_results.png")


if __name__ == "__main__":
    run_evaluation()