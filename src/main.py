import math
import torch
import numpy as np
import tqdm
from datasets.swissmetro_dataset import SwissmetroDataSet
from models.model_MLP_2 import MLP2Model16, MLP2Model2
from models.model_MNL import MNLModel
from models.model_MLP import MLPModel16, MLPModel8, MLPModel2
from models.model_reslogit import ResLogitModel
from models.model_LMNL import LMNLModel
from torch_utils import compute_accuracy, log_likelihood, plot_LL, plot_loss, save_models

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(4328) # important pour les tests

# Preparation données
dataset = SwissmetroDataSet("data/swissmetro.dat")
train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=[0.5, 0.5])
batch_size = 50
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffle important
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Initialisation modèle
models = {
    'mnl': MNLModel(), # MultiNomial Logit
    'mlp-2': MLPModel2(), # MLP network with 2 hidden layers
    'mlp-16': MLPModel16(), # MLP network with 16 hidden layers
    'mlp_2-2': MLP2Model2(), # MLP network with 2 hidden layers, BatchNormalization, Non-Linearity (ReLu) and Dropout
    'mlp_2-16': MLP2Model16(), # MLP network with 16 hidden layers, BatchNormalization, Non-Linearity (ReLu) and Dropout
    'reslogit-2': ResLogitModel(2), # ResLogit model with 2 residual layers
    'reslogit-16': ResLogitModel(16), # ResLogit model with 16 residual layers
    'lmnl-16': LMNLModel() # Learning MultiNomial Logit
}

models_loss = {}
models_accuracy_test = {}
models_accuracy_train = {}
models_LL = {}

# Initialisation rétropropagation
learning_rate = 0.001 # pour réduire l'oscillation de la trajectoire de la fonction de perte
loss_fct = torch.nn.CrossEntropyLoss()

# Lancement entrainement
n_epochs = 600
n_batches_per_epoch = int(math.ceil(len(train_loader.dataset) / batch_size))

for model_name, model in models.items():
    model.to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    test_best_accuracy = 0
    train_best_accuracy = 0
    test_best_weights = None
    train_best_weights = None
    test_accuracy_per_epoch = []
    train_accuracy_per_epoch = []
    loss_per_epoch = []
    log_likelihood_per_epoch = []

    for t in range(n_epochs):
        model.train()
        with tqdm.trange(n_batches_per_epoch, unit='batch') as epoch_bar:
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_log_likelihood = 0
            epoch_bar.set_description(desc=f'Epoch #{t}')
            
            for i, load_target in zip(epoch_bar, train_loader):
                data, targets = load_target
                data = data.to(DEVICE)
                targets = targets.to(DEVICE)
                prediction = model(data)
                optimizer.zero_grad()
                loss = loss_fct(prediction, targets)
                loss.backward()
                optimizer.step()
                pred_targets = torch.argmax(prediction, dim=1)

                epoch_loss += float(loss)
                epoch_accuracy += float(torch.count_nonzero(pred_targets == targets) / len(targets))
                epoch_log_likelihood += float(log_likelihood(prediction, targets))

                one_indexed_i = i + 1
                epoch_bar.set_postfix({
                    'loss': epoch_loss / one_indexed_i,
                    'accuracy': epoch_accuracy / one_indexed_i,
                    'log_likelihood': epoch_log_likelihood / one_indexed_i
                })

        # Utile pour savoir à quelle epoch le modèle a le mieux performé
        epoch_test_accuracy = compute_accuracy(model, test_loader)
        epoch_train_accuracy = compute_accuracy(model, train_loader)
        loss_per_epoch.append(epoch_loss)
        test_accuracy_per_epoch.append(epoch_test_accuracy)
        train_accuracy_per_epoch.append(epoch_train_accuracy)
        log_likelihood_per_epoch.append(epoch_log_likelihood)

        if epoch_test_accuracy > test_best_accuracy:
            test_best_accuracy = epoch_test_accuracy
            test_best_weights = model.state_dict()
        if epoch_train_accuracy > train_best_accuracy:
            train_best_accuracy = epoch_train_accuracy
            train_best_weights = model.state_dict()
        
        print(f'Epoch {t} test accuracy: {epoch_test_accuracy * 100}%, train accuracy: {epoch_train_accuracy * 100}%')

    print(f'Meilleur cas test accuracy: {test_best_accuracy * 100}%, train accuracy: {train_best_accuracy * 100}%')

    models_loss[model_name] = loss_per_epoch
    models_LL[model_name] = log_likelihood_per_epoch
    models_accuracy_test[model_name] = test_best_accuracy
    models_accuracy_train[model_name] = train_best_accuracy

    concatenated_data = np.column_stack((loss_per_epoch, log_likelihood_per_epoch))
    np.savetxt(f'./generated_data/{model_name}_data.csv', concatenated_data, delimiter=';')
    
    save_models(model, model_name)

print("Best accuracies test:" + str(models_accuracy_test))
print("Best accuracies train:" + str(models_accuracy_train))

plot_loss(n_epochs, models_loss)
plot_LL(n_epochs, models_LL)