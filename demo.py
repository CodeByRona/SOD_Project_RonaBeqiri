"""
demo.py - Single-image inference demo for Salient Object Detection.

Displays four panels:
  1. Input Image
  2. Predicted Saliency Map (heatmap)
  3. Binary Mask (threshold 0.5)
  4. Overlay (saliency mask blended onto original image)

Also prints inference time per image.
"""

import time
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from sod_model import SODModel


def run_demo(image_path=None):
    """
    Run inference on a single image and display a 4-panel visualization.

    Args:
        image_path (str): Path to input image. If None, uses the last image
                          in dataset/images/ as a default test case.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────────
    model = SODModel().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # ── Select image ──────────────────────────────────────────────────────────
    if image_path is None:
        img_folder = "dataset/images"
        image_path = os.path.join(img_folder, sorted(os.listdir(img_folder))[-1])

    print(f"Running inference on: {image_path}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    original_img = Image.open(image_path).convert("RGB")
    original_resized = original_img.resize((128, 128))

    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    img_tensor = transform(original_img).unsqueeze(0).to(device)

    # ── Inference (with timing) ───────────────────────────────────────────────
    start_time = time.time()
    with torch.no_grad():
        pred = model(img_tensor)
    inference_time = time.time() - start_time

    print(f"Inference time: {inference_time * 1000:.1f} ms")

    # ── Post-process outputs ──────────────────────────────────────────────────
    saliency_map = pred.squeeze().cpu().numpy()          # continuous [0, 1]
    binary_mask = (saliency_map > 0.5).astype(np.float32)  # thresholded

    # Overlay: blend the binary mask (red channel) onto the original image
    img_array = np.array(original_resized).astype(np.float32) / 255.0
    overlay = img_array.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + 0.5 * binary_mask, 0, 1)  # boost red
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] - 0.3 * binary_mask, 0, 1)  # suppress green
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] - 0.3 * binary_mask, 0, 1)  # suppress blue

    # ── Plot 4-panel figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Salient Object Detection  |  Inference time: {inference_time * 1000:.1f} ms", fontsize=13)

    axes[0].imshow(original_resized)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(saliency_map, cmap='jet')
    axes[1].set_title("Predicted Saliency")
    axes[1].axis("off")

    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title("Binary Mask (threshold 0.5)")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (mask on image)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig("demo_output.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualization saved to demo_output.png")


if __name__ == "__main__":
    run_demo()