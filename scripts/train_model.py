"""Train ResNet-18 on CIFAR-10 for a given colorspace (RGB / HSV / LAB).

Usage:
    python -m scripts.train_model --config configs/baseline_rgb.yaml
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import yaml

from color_dg.data.loader import get_cifar10_loaders
from color_dg.models.resnet import create_resnet18


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()  # active dropout, BatchNorm en mode train
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()          # rsets the gradients to zero
        outputs = model(inputs)        # forward pass
        loss = criterion(outputs, targets)  # calculation based on loss
        loss.backward()                # backward pass (calculation of gradients)
        optimizer.step()               # update weights

        total_loss += loss.item() * inputs.size(0)  # cumulative loss
        correct += outputs.argmax(1).eq(targets).sum().item()  # correct predictions
        total += inputs.size(0)        # total number of examples seen

    return total_loss / total, correct / total  # average loss, accuracy



def eval_epoch(model, loader, criterion, device):
    model.eval()  # BatchNorm uses running statistics, not batch statistics
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():  # disables gradient calculation
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total  # average loss, accuracy





def main():
    # --- 1. Command-line arguments---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the YAML file")
    args = parser.parse_args()

    # --- 2. Loading the YAML configuration ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- 3. Device (GPU if available, otherwise CPU) ---
    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    print(f"[{cfg['experiment']}] device={device}  colorspace={cfg['colorspace']}")

    # --- 4. output directory---
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # --- 5. Data loaders ---
    train_loader, test_loader = get_cifar10_loaders(
        colorspace=cfg["colorspace"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # --- 6. models ---
    model = create_resnet18(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"].get("pretrained", False),
    ).to(device)

    # --- 7. Loss, optimizer, scheduler ---
    criterion = nn.CrossEntropyLoss()

    if cfg["training"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["training"]["learning_rate"],
                                    momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"]["scheduler_params"]["step_size"],
        gamma=cfg["training"]["scheduler_params"]["gamma"],
    )



    # --- 8. training loop ---
    best_acc = 0.0
    rows = []

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, test_loader,  criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{cfg['training']['epochs']} "
            f"| train_loss={train_loss:.4f}  train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f}  val_acc={val_acc:.4f} "
            f"| {elapsed:.1f}s"
        )

        rows.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss,     "val_acc": val_acc,
        })

        # Saving the best template
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(cfg["output_dir"], "best_model.pth"))

    # --- 9. Saving metrics ---
    csv_path = os.path.join(cfg["output_dir"], "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment":    cfg["experiment"],
        "colorspace":    cfg["colorspace"],
        "best_val_acc":  best_acc,
    }
    with open(os.path.join(cfg["output_dir"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest val accuracy : {best_acc:.4f}")
    print(f"Results saved to  : {cfg['output_dir']}")


if __name__ == "__main__":
    main()
