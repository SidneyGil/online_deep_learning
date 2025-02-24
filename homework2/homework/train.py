import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .utils import load_data, compute_accuracy


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create log directory with timestamp
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load the model
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Load dataset (set num_workers=0 for Colab)
    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=0)
    val_data = load_data("classification_data/val", shuffle=False, num_workers=0)

    # Define loss function & optimizer
    loss_func = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # Training loop
    for epoch in range(num_epoch):
        metrics["train_acc"].clear()
        metrics["val_acc"].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # Forward pass
            pred = model(img)
            loss_val = loss_func(pred, label)

            # Backpropagation
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store training accuracy
            metrics['train_acc'].append(compute_accuracy(pred, label))

            global_step += 1

        # Evaluation loop (disable gradients)
        with torch.inference_mode():
            model.eval()
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                metrics['val_acc'].append(compute_accuracy(pred, label))

        # Compute mean accuracy for epoch
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        # logging key names (match logger.py)
        logger.add_scalar('train_accuracy', epoch_train_acc, global_step)
        logger.add_scalar('val_accuracy', epoch_val_acc, global_step)

        # Print every 10 epochs + first & last
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # Save and overwrite the model for grading
    save_model(model)

    # Save model to log directory for backup
    model_path = log_dir / f"{model_name}.th"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Confirm model exists for grading
    if not model_path.exists():
        raise RuntimeError(f"Model file {model_path} not saved correctly!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # Pass all arguments to train
    train(**vars(parser.parse_args()))
