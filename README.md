# End-to-End Salient Object Detection (SOD)

A complete Salient Object Detection pipeline built from scratch using PyTorch — no pre-trained weights.  
Trained on the **ECSSD dataset** (1,000 images) with a custom CNN Encoder-Decoder architecture.

---

## Results

| Metric         | Score         |
| -------------- | ------------- |
| Mean IoU       | 0.2918        |
| Mean Precision | 0.5530        |
| Mean Recall    | 0.4291        |
| Mean F1-Score  | 0.4202        |
| Inference Time | 28.0 ms (CPU) |

---

## Project Structure

```
SOD_Project/
├── data_loader.py       # Dataset loading, preprocessing, and augmentation
├── sod_model.py         # CNN Encoder-Decoder architecture
├── train.py             # Custom training loop with validation and checkpointing
├── evaluate.py          # Evaluation metrics (IoU, Precision, Recall, F1)
├── demo.py              # Single-image inference with 4-panel visualization
├── best_model.pth       # Saved weights from best training checkpoint (epoch 11)
└── README.md
```

---

## Dataset

This project uses the **ECSSD (Extended Complex Scene Saliency Dataset)**.

- 1,000 semantically complex images with pixel-accurate saliency masks
- Download: [https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)

After downloading, organize the dataset as follows:

```
SOD_Project/
└── dataset/
    ├── images/      # .jpg input images
    └── masks/       # .png ground-truth saliency masks
```

---

## Model Architecture

Custom CNN Encoder-Decoder built entirely from scratch:

**Encoder** — 3 convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool):

- Block 1: 3 → 32 channels | 128×128 → 64×64
- Block 2: 32 → 64 channels | 64×64 → 32×32
- Block 3: 64 → 128 channels | 32×32 → 16×16 _(+ Dropout2d 0.3)_

**Decoder** — 3 transposed convolution blocks (ConvTranspose2D → BatchNorm → ReLU):

- Block 1: 128 → 64 channels | 16×16 → 32×32
- Block 2: 64 → 32 channels | 32×32 → 64×64
- Block 3: 32 → 16 channels | 64×64 → 128×128

**Output** — 1×1 Conv + Sigmoid → saliency probability map (128×128)

---

## Installation

```bash
pip install torch torchvision matplotlib pillow tqdm numpy
```

## Usage

### 1. Train the model

```bash
python train.py
```

- Trains for up to 30 epochs with early stopping (patience = 5)
- Saves `best_model.pth` based on best validation loss
- Saves `checkpoint.pth` after every epoch for resume support
- If training is interrupted, re-running `train.py` automatically resumes from the last checkpoint

### 2. Evaluate on the test set

```bash
python evaluate.py
```

Prints Mean IoU, Precision, Recall, and F1-Score.  
Saves a visualization grid to `evaluation_results.png`.

### 3. Run the demo

```bash
python demo.py
```

Runs inference on a single image and displays a 4-panel visualization:

- Input Image
- Predicted Saliency Map (heatmap)
- Binary Mask (threshold 0.5)
- Overlay (mask blended onto original image)

Also prints inference time in milliseconds.

---

## Training Details

| Setting         | Value                       |
| --------------- | --------------------------- |
| Loss Function   | BCE + 0.5 x (1 - IoU)       |
| Optimizer       | Adam, lr = 1e-3             |
| Batch Size      | 16                          |
| Max Epochs      | 30                          |
| Early Stopping  | Patience = 5                |
| Best Checkpoint | Epoch 11, Val Loss = 0.8519 |
| Hardware        | CPU                         |

---
