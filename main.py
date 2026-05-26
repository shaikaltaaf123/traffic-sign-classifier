# main.py
# Entry point for the traffic sign classification project.
# Run this file to train and evaluate the model:
#   python main.py

import os
import torch

from src.dataset import get_dataloaders
from src.model import TrafficSignNet, get_model_summary
from src.train import train_model
from src.evaluate import (evaluate_model, print_classification_report,
                          plot_confusion_matrix, plot_training_curves)
from src.utils import get_device, set_seed

# ── Configuration ─────────────────────────────────────────────────
# Change these values to experiment with different settings
DATA_DIR = "./data"
SAVE_DIR = "./models"
PLOTS_DIR = "./plots"
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
SEED = 42
# ──────────────────────────────────────────────────────────────────


def main():
    print()
    print("=" * 62)
    print("  Traffic Sign Classification — GTSRB")
    print("=" * 62)
    print()

    # Reproducibility — same seed = same results every run
    set_seed(SEED)

    # Detect best available device
    device = get_device()
    print()

    # Load dataset
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )

    # Build model and move to device
    model = TrafficSignNet(num_classes=num_classes).to(device)
    get_model_summary(model)

    # Train — saves best model automatically to SAVE_DIR/best_model.pth
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        save_dir=SAVE_DIR,
    )

    # Load the best checkpoint for final evaluation
    # (best val accuracy, not necessarily the last epoch)
    best_path = os.path.join(SAVE_DIR, "best_model.pth")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']}")
    print(f"Best val accuracy was: {checkpoint['val_acc']*100:.2f}%\n")

    # Evaluate on test set — completely unseen during training
    results = evaluate_model(model, test_loader, device)

    # Per-class precision, recall, F1
    print_classification_report(results['all_labels'], results['all_preds'])

    # Save plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_training_curves(
        history,
        save_path=os.path.join(PLOTS_DIR, "training_curves.png")
    )
    plot_confusion_matrix(
        results['all_labels'],
        results['all_preds'],
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png")
    )

    # Final summary
    print()
    print("=" * 62)
    print("  RESULTS SUMMARY")
    print("=" * 62)
    print(f"  Dataset  : GTSRB — German Traffic Sign Recognition Benchmark")
    print(f"  Classes  : {num_classes}")
    print(f"  Test Acc : {results['accuracy']*100:.2f}%")
    print(f"  Model    : TrafficSignNet (custom CNN, PyTorch)")
    print(f"  Plots    : {PLOTS_DIR}/")
    print(f"  Weights  : {best_path}")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
