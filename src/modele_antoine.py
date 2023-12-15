import torch
import numpy as np
import tqdm
import torch.nn.functional as F
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

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ASC_TRAIN = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_SM = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_CAR = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_TIME = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_COST = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)

        self.linear1 = torch.nn.Linear(30, 120) # 27 variables en entrée
        self.linear2 = torch.nn.Linear(120, 60) # 3 classes de sortie
        self.linear3 = torch.nn.Linear(60, 3) # 3 classes de sortie

        self.relu = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):
        V1 = self.ASC_TRAIN + self.B_TIME * x[:, 19] + self.B_COST * x[:, 20]
        V1 = V1.unsqueeze(1)
        V2 = self.ASC_SM + self.B_TIME * x[:, 22] + self.B_COST * x[:, 23]
        V2 = V2.unsqueeze(1)
        V3 = self.ASC_CAR + self.B_TIME * x[:, 25] + self.B_COST * x[:, 26]
        V3 = V3.unsqueeze(1) # marche pas
        # V = torch.stack((V1, V2, V3), dim=1)
        V = torch.stack((x, V1, V2, V3), dim=1)

        y = self.dropout1(self.relu(self.linear1(V)))
        y = self.dropout2(self.relu(self.linear2(y)))

        # probabilities = F.softmax(V, dim=1)
        # return probabilities
    
        return self.linear3(y)

if __name__ == '__main__':
    # Preparation données
    dataset = SwissmetroDataSet("data/swissmetro.dat")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=[0.5, 0.5])
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Initialisation modèle
    model = Model1()
    # model = SwissmetroModel()
    model.to(DEVICE)

    # Initialisation rétropropagation
    learning_rate = 0.01
    momentum = 0.2
    loss_fct = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Lancement entrainement
    n_epochs = 1000
    n_batches_per_epoch = len(train_dataset) // batch_size + 1
    for t in range(n_epochs):
        model.train()
        with tqdm.trange(n_batches_per_epoch, unit='batch') as epoch_bar:
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_bar.set_description(desc=f'Epoch #{t}')
            for i, batch in zip(epoch_bar, train_loader):
                data, targets = batch
                data = data.to(DEVICE)
                targets = targets.to(DEVICE)
                prediction = model(data)  # forward
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