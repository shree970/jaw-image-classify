from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.test import test
from utils.train import train


def trainer(model, epochs, device, train_loader, test_loader, optimizer, criterion, l1_factor, use_scheduler=True):
    """
    Train and evaluate for given epochs.
    """
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    lrs = []
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3, verbose=True, mode="max")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}:")
        train(model, device, train_loader, optimizer, train_accuracy, train_losses, l1_factor, criterion, lrs)
        test(model, device, test_loader, test_accuracy, test_losses, criterion)

        if use_scheduler:
            scheduler.step(test_accuracy[-1])

    return train_accuracy, train_losses, test_accuracy, test_losses
