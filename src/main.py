import math
import torch
import numpy as np
import tqdm
from datasets.swissmetro_dataset import SwissmetroDataSet
from models.model_multinomial_logit import MultinomialLogitModel
from models.model_mlp import MLPModel16, MLPModel2, MLPModel8
from models.model_reslogit import ResLogitModel
from models.model_LMNL import LMNLModel
from torch_utils import compute_accuracy

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
    'mnl': MultinomialLogitModel(),
    'mlp-2': MLPModel2(),
    'mlp-8': MLPModel8(),
    'mlp-16': MLPModel16(),
    'reslogit-2': ResLogitModel(2),
    'reslogit-8': ResLogitModel(8),
    'reslogit-16': ResLogitModel(16),
    'lmnl': LMNLModel()
}
model = models['mlp-2']
model.to(DEVICE)

# Initialisation rétropropagation
learning_rate = 0.01
loss_fct = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# Lancement entrainement
n_epochs = 600
n_batches_per_epoch = int(math.ceil(len(train_loader.dataset) / batch_size))
best_accuracy = 0
best_weights = None
accuracy_per_epoch = []
loss_per_epoch = []
for t in range(n_epochs):
    model.train()
    with tqdm.trange(n_batches_per_epoch, unit='batch') as epoch_bar:
        epoch_loss = 0
        epoch_accuracy = 0
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
            with torch.no_grad():
                max_likelihood = torch.softmax(prediction, dim=1)
            pred_targets = torch.argmax(prediction, dim=1)
            epoch_accuracy += float(torch.count_nonzero(pred_targets == targets) / len(targets))
            epoch_loss += float(loss)

            one_indexed_i = i + 1
            epoch_bar.set_postfix({
                'loss': epoch_loss / one_indexed_i,
                'accuracy': epoch_accuracy / one_indexed_i
            })

    # Utile pour savoir à quelle epoch le modèle a le mieux performé
    epoch_test_accuracy = compute_accuracy(model, test_loader)
    loss_per_epoch.append(epoch_loss)
    accuracy_per_epoch.append(epoch_test_accuracy)
    if epoch_test_accuracy > best_accuracy:
        best_accuracy = epoch_test_accuracy
        best_weights = model.state_dict()
    print(f'Epoch {t} test accuracy: {epoch_test_accuracy * 100}%')

print(f'Meilleur cas: {best_accuracy * 100}%')

for name, modelObj in models.items():
    if modelObj == model:
        model_name = name 

np.savetxt(f'{model_name}_test_accuracy.csv', np.array(accuracy_per_epoch), delimiter=';')
np.savetxt(f'{model_name}_loss.csv', np.array(loss_per_epoch), delimiter=';')