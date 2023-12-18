from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

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

def log_likelihood(predictions, targets):
    # Supposons que vos prédictions suivent une distribution de probabilité, 
    # par exemple, à travers une fonction softmax.
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    
    # Sélectionnez la probabilité associée à la classe correcte.
    correct_probabilities = torch.gather(probabilities, 1, targets.view(-1, 1))
    
    # Calculez le log de la probabilité correcte.
    log_probs = torch.log(correct_probabilities)
    
    # Sommez les log-vraisemblances pour chaque exemple dans le lot.
    return torch.sum(log_probs)

def plot_loss(epochs, loss_models):
    plt.figure(figsize=(12, 6))

    # Tracer les courbes d'entraînement et de test
    for model_name, loss in loss_models.items():
        plt.plot(list(range(epochs)), loss, label=model_name, linestyle='-')

    # Ajouter des étiquettes et une légende
    plt.xlabel('Nombre d\'époques', fontsize=20)
    plt.xticks(fontsize=16)
    plt.ylabel('Loss', fontsize=20)
    plt.yticks(fontsize=16)
    plt.title('Loss des différents modèles', fontsize=20)
    plt.legend(fontsize=16, loc='upper right', frameon=False)

    # Afficher la grille pour une meilleure lisibilité
    plt.grid(True, linestyle='--', alpha=0.7)

    # Afficher le graphique
    plt.show()
        
def plot_LL(epochs, likehood_models):
    plt.figure(figsize=(12, 6))

    # Tracer les courbes d'entraînement et de test
    for model_name, likehood in likehood_models.items():
        plt.plot(list(range(epochs)), likehood, label=model_name, linestyle='-')

    # Ajouter des étiquettes et une légende
    plt.xlabel('Nombre d\'époques', fontsize=20)
    plt.xticks(fontsize=16)
    plt.ylabel('Log-vraisemblance', fontsize=20)
    plt.yticks(fontsize=16)
    plt.title('Log-vraisemblance des différents modèles', fontsize=20)
    plt.legend(fontsize=16, loc='lower right', frameon=False)

    # Afficher la grille pour une meilleure lisibilité
    plt.grid(True, linestyle='--', alpha=0.7)

    # Afficher le graphique
    plt.show()

def save_models(model, name):
    MODEL_PATH = Path("generated_models")
    MODEL_PATH.mkdir(parents=True,exist_ok=True)

    MODEL_NAME = name + ".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model state dict at : {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)