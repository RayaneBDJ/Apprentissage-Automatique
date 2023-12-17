import math
import torch
import numpy as np
import tqdm
from datasets.swissmetro_dataset import SwissmetroDataSet
from model import SwissMetroResLogit
from torch_utils import compute_accuracy

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(4328)
# Preparation données
dataset = SwissmetroDataSet("data/swissmetro.dat")
train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=[0.5, 0.5])
batch_size = 50
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Initialisation modèle
model = SwissMetroResLogit()
model.to(DEVICE)

# Initialisation rétropropagation
learning_rate = 0.01
momentum = 0.2
loss_fct = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# Lancement entrainement
n_epochs = 1000
n_batches_per_epoch = int(math.ceil(len(train_loader.dataset) / batch_size))
best_accuracy = 0
best_weights = None
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
            pred_targets = torch.argmax(prediction, dim=1)
            epoch_accuracy += float(torch.count_nonzero(pred_targets == targets) / len(targets))
            epoch_loss += float(loss)

            one_indexed_i = i + 1
            epoch_bar.set_postfix({
                'loss': epoch_loss / one_indexed_i,
                'accuracy': epoch_accuracy / one_indexed_i
            })
    # Utile pour savoir à quelle epoch le modèle a le mieux performé
    epoch_accuracy = compute_accuracy(model, test_loader)
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        best_weights = model.state_dict()
    print(f'Epoch {t} test accuracy: {epoch_accuracy* 100}%')

print(f'Meilleur cas: {best_accuracy * 100}%')