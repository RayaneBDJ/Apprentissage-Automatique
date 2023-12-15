
import numpy as np
import torch

def compute_accuracy(model, dataloader, device='cpu') -> float:
    model.eval()
    all_predictions = []
    all_targets = []
    
    for batch in dataloader:
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            predictions = model(images)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    if all_predictions[0].shape[-1] > 1:
        predictions_numpy = np.concatenate(all_predictions, axis=0)
        predictions_numpy = predictions_numpy.argmax(axis=1)
        targets_numpy = np.concatenate(all_targets, axis=0)
    else:
        predictions_numpy = np.concatenate(all_predictions).squeeze(-1)
        targets_numpy = np.concatenate(all_targets)
        predictions_numpy[predictions_numpy >= 0.5] = 1.0
        predictions_numpy[predictions_numpy < 0.5] = 0.0
    
    return (predictions_numpy == targets_numpy).mean()
