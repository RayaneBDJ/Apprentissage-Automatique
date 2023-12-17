import torch

class MLPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(27, 270)
        self.linear2 = torch.nn.Linear(270, 135)
        self.linear3 = torch.nn.Linear(135, 70)
        self.linear4 = torch.nn.Linear(70, 70)
        self.linear5 = torch.nn.Linear(70, 30)
        self.linear6 = torch.nn.Linear(30, 3)

        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.3)

    def forward(self, x):
        y = self.dropout1(self.relu(self.linear1(x)))
        y = self.dropout2(self.relu(self.linear2(y)))
        y = self.dropout2(self.relu(self.linear3(y)))
        y = self.dropout2(self.relu(self.linear4(y)))
        y = self.relu(self.linear5(y))

        return self.linear6(y) # Pas de fonction d'activation car CrossEntropyLoss s'en charge pour nous