import torch
import numpy as np
import tqdm
from datasets.swissmetro_dataset import SwissmetroDataSet
from torch_utils import compute_accuracy

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SwissmetroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Choix du nombre de couches arbitraire. Jouer avec.
        self.linear1 = torch.nn.Linear(27, 1080) # 27 variables en entrée
        self.linear2 = torch.nn.Linear(1080, 540) # 3 classes de sortie
        self.linear3 = torch.nn.Linear(540, 3) # 3 classes de sortie

        self.relu = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):
        y = self.dropout1(self.relu(self.linear1(x)))
        y = self.dropout2(self.relu(self.linear2(y)))

        return self.linear3(y) # Pas de fonction d'activation car CrossEntropyLoss s'en charge pour nous


if __name__ == '__main__':
    # Preparation données
    dataset = SwissmetroDataSet("data/swissmetro.dat")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=[0.5, 0.5])
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Initialisation modèle
    model = SwissmetroModel()
    model.to(DEVICE)

    # Initialisation rétropropagation
    learning_rate = 0.01
    momentum = 0.2
    loss_fct = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    # Lancement entrainement
    n_epochs = 1000
    n_batches_per_epoch = len(train_dataset) // batch_size + 1
    for t in range(n_epochs):
        model.eval()
        with tqdm.trange(n_batches_per_epoch, unit='batch') as epoch_bar:
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_bar.set_description(desc=f'Epoch #{t}')
            for i, batch in zip(epoch_bar, train_loader):
                data, targets = batch
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
        print(f'Epoch {t} test accuracy: {compute_accuracy(model, test_loader)}')