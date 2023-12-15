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
        self.linear1 = torch.nn.Linear(27, 540) # 27 variables en entrée
        self.linear2 = torch.nn.Linear(540, 72) # 3 classes de sortie
        self.linear3 = torch.nn.Linear(72, 3) # 3 classes de sortie
        self.linear4 = torch.nn.Linear(3, 3) 
        # self.linear7.weight = torch.nn.Parameter(
        #     torch.from_numpy(np.array(
        #         [[-0.701187, 0., -0.154633],
        #         [0., 0.701187, 0.546555],
        #         [-0.546555, 0.154633, 0.]]).astype(np.float32)), 
        #      requires_grad=True)
        # self.linear7.bias = torch.nn.Parameter(
        #     torch.from_numpy(np.array([-0.415914, 0.285273, 0.130641]).astype(np.float32)), 
        #      requires_grad=True)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.25)
        self.batch_norm1 = torch.nn.BatchNorm1d(540)
        self.batch_norm2 = torch.nn.BatchNorm1d(72)
        self.batch_norm3 = torch.nn.BatchNorm1d(3)

    def forward(self, x):
        y = self.batch_norm1(self.dropout(self.relu(self.linear1(x))))
        y = self.batch_norm2(self.dropout(self.relu(self.linear2(y))))
        y = self.batch_norm3(self.dropout(self.relu(self.linear3(y))))

        return self.linear4(y) # Pas de fonction d'activation car CrossEntropyLoss s'en charge pour nous


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