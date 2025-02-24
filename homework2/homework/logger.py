from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    global_step = 0
    for epoch in range(10):
        metrics = {"train_acc": [], "val_acc": []}

        # Training Loop
        torch.manual_seed(epoch)  # ✅ Ensures reproducibility
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(10)  # Generates the same numbers per epoch

            # ✅ Log training loss with correct step
            logger.add_scalar('train_loss', dummy_train_loss, global_step)

            # ✅ Append accuracy for averaging
            metrics["train_acc"].append(dummy_train_accuracy.mean().item())

            global_step += 1  # Make sure global_step increases correctly

        # ✅ Log train accuracy once per epoch (use global_step for consistency)
        avg_train_acc = sum(metrics["train_acc"]) / len(metrics["train_acc"])
        logger.add_scalar('train_accuracy', avg_train_acc, global_step)

        # Validation Loop
        torch.manual_seed(epoch)  # ✅ Set seed again for validation
        for _ in range(10):
            dummy_validation_accuracy = epoch / 10.0 + torch.randn(10)
            metrics["val_acc"].append(dummy_validation_accuracy.mean().item())

        # ✅ Log validation accuracy once per epoch (use global_step for consistency)
        avg_val_acc = sum(metrics["val_acc"]) / len(metrics["val_acc"])
        logger.add_scalar('val_accuracy', avg_val_acc, global_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)
