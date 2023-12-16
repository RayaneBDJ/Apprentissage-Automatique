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

    def forward(self, x):
        V1 = self.ASC_TRAIN + self.B_TIME * x[:, 19] + self.B_COST * x[:, 20]
        V2 = self.ASC_SM + self.B_TIME * x[:, 22] + self.B_COST * x[:, 23]
        V3 = self.ASC_CAR + self.B_TIME * x[:, 25] + self.B_COST * x[:, 26]
        y = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,24]] # enlève col TRAIN [19, 20], SM [22, 23], CAR [25, 26]
        V = torch.concat((y, V1.unsqueeze(1), V2.unsqueeze(1), V3.unsqueeze(1)), dim=1)

        return V
    
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ASC_TRAIN = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_SM = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_CAR = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_TIME = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_COST = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)

        self.linear1 = torch.nn.Linear(24, 120) # 27 variables en entrée
        self.linear2 = torch.nn.Linear(120, 60) # 3 classes de sortie
        self.linear3 = torch.nn.Linear(60, 3) # 3 classes de sortie

        self.relu = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):
        V1 = self.ASC_TRAIN + self.B_TIME * x[:, 19] + self.B_COST * x[:, 20]
        V2 = self.ASC_SM + self.B_TIME * x[:, 22] + self.B_COST * x[:, 23]
        V3 = self.ASC_CAR + self.B_TIME * x[:, 25] + self.B_COST * x[:, 26]
        y = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,24]] # enlève col TRAIN [19, 20], SM [22, 23], CAR [25, 26]

        V = torch.concat((y, V1.unsqueeze(1), V2.unsqueeze(1), V3.unsqueeze(1)), dim=1)

        y = self.dropout1(self.relu(self.linear1(V)))
        y = self.dropout2(self.relu(self.linear2(y)))
    
        return self.linear3(y)

 
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ASC_TRAIN = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_SM = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_CAR = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_TIME = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_COST = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)

        self.theta_parameters = torch.nn.ParameterList()
        self.theta_parameter1 = torch.nn.Parameter(torch.randn(24,24), requires_grad=True)
        self.theta_parameter2 = torch.nn.Parameter(torch.randn(24,24), requires_grad=True)
        self.theta_parameters.append(self.theta_parameter1)
        self.theta_parameters.append(self.theta_parameter2)

        self.linear1 = torch.nn.Linear(24, 120) # 27 variables en entrée
        self.linear2 = torch.nn.Linear(120, 60) # 3 classes de sortie
        self.linear3 = torch.nn.Linear(60, 3) # 3 classes de sortie

        self.relu = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):
        # residual = x

        V1 = self.ASC_TRAIN + self.B_TIME * x[:, 19] + self.B_COST * x[:, 20]
        V2 = self.ASC_SM + self.B_TIME * x[:, 22] + self.B_COST * x[:, 23]
        V3 = self.ASC_CAR + self.B_TIME * x[:, 25] + self.B_COST * x[:, 26]
        y = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,24]] # enlève col TRAIN [19, 20], SM [22, 23], CAR [25, 26]

        V = torch.concat((y, V1.unsqueeze(1), V2.unsqueeze(1), V3.unsqueeze(1)), dim=1)

        out = self.linear1(V)
        out -= F.softplus(torch.matmul(out, self.theta_parameter1))
        out = self.linear2(out)
        out += V

        out = self.linear3(out)
        
        return V

        # y = self.dropout1(self.relu(self.linear1(V)))
        # y = self.dropout2(self.relu(self.linear2(y)))

        # sum = 0
        # for theta_parameter in self.theta_parameters:
        #     V1 = F.softplus(torch.matmul(V1, self.theta_parameter))
        #     V2 = torch.nn.functional.softplus(V2 + torch.matmul(theta_parameter, V2))
        #     V3 = torch.nn.functional.softplus(V3 + torch.matmul(theta_parameter, V3))
        #     new_V = F.softplus(torch.matmul())

        # x += residual - sum
    
        # return self.linear3(y)

# class ResidualBlock(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(ResidualBlock, self).__init__()

#         # Couche entièrement connectée 1
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.bn1 = nn.BatchNorm1d(hidden_size)
#         self.relu = nn.ReLU()

#         # Couche entièrement connectée 2
#         self.fc2 = nn.Linear(hidden_size, input_size)
#         self.bn2 = nn.BatchNorm1d(input_size)

#     def forward(self, x):
#         # Passage à travers la première couche entièrement connectée
#         out = self.fc1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         # Passage à travers la deuxième couche entièrement connectée
#         out = self.fc2(out)
#         out = self.bn2(out)

#         # Connexion résiduelle
#         out += x
#         out = self.relu(out)

#         return out


if __name__ == '__main__':
    # Preparation données
    dataset = SwissmetroDataSet("data/swissmetro.dat")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=[0.5, 0.5])
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Initialisation modèle
    model = Model3()
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