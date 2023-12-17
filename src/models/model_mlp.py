import torch


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MLPModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input = torch.nn.Linear(27, 270) # 27 variables en entrée
        self.output = torch.nn.Linear(270, 3) # 3 classes de sortie

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        y = self.dropout(self.relu(self.input(x)))

        return self.output(y)
    
class MLPModel8(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input = torch.nn.Linear(27, 270) # 27 variables en entrée
        self.layer1 = torch.nn.Linear(270, 200)
        self.layer2 = torch.nn.Linear(200, 140)
        self.layer3 = torch.nn.Linear(140, 100)
        self.layer4 = torch.nn.Linear(100, 60)
        self.layer5 = torch.nn.Linear(60, 45)
        self.layer6 = torch.nn.Linear(45, 30)
        self.output = torch.nn.Linear(30, 3) # 3 classes de sortie

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        y = self.dropout(self.relu(self.input(x)))
        y = self.dropout(self.relu(self.layer1(y)))
        y = self.dropout(self.relu(self.layer2(y)))
        y = self.dropout(self.relu(self.layer3(y)))
        y = self.dropout(self.relu(self.layer4(y)))
        y = self.dropout(self.relu(self.layer5(y)))
        y = self.dropout(self.relu(self.layer6(y)))

        return self.output(y)
    
class MLPModel16(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input = torch.nn.Linear(27, 135) # 27 variables en entrée
        self.layer1 = torch.nn.Linear(135, 70)
        self.layer2 = torch.nn.Linear(70, 60)
        self.layer3 = torch.nn.Linear(60, 50)
        self.layer4 = torch.nn.Linear(50, 45)
        self.layer5 = torch.nn.Linear(45, 40)
        self.layer6 = torch.nn.Linear(40, 60)
        self.layer7 = torch.nn.Linear(60, 55)
        self.layer8 = torch.nn.Linear(55, 50)
        self.layer9 = torch.nn.Linear(50, 45)
        self.layer10 = torch.nn.Linear(45, 40)
        self.layer11 = torch.nn.Linear(40, 30)
        self.layer12 = torch.nn.Linear(30, 25)
        self.layer13 = torch.nn.Linear(25, 20)
        self.layer14 = torch.nn.Linear(20, 10)
        self.output = torch.nn.Linear(10, 3) # 3 classes de sortie

        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.1)

    def forward(self, x):
        y = self.dropout1(self.relu(self.input(x)))
        y = self.dropout1(self.relu(self.layer1(y)))
        y = self.dropout1(self.relu(self.layer2(y)))
        y = self.dropout1(self.relu(self.layer3(y)))
        y = self.dropout1(self.relu(self.layer4(y)))
        y = self.dropout1(self.relu(self.layer5(y)))
        y = self.dropout1(self.relu(self.layer6(y)))
        y = self.dropout1(self.relu(self.layer7(y)))
        y = self.dropout1(self.relu(self.layer8(y)))
        y = self.dropout1(self.relu(self.layer9(y)))
        y = self.dropout1(self.relu(self.layer10(y)))
        y = self.dropout2(self.relu(self.layer11(y)))
        y = self.dropout2(self.relu(self.layer12(y)))
        y = self.dropout2(self.relu(self.layer13(y)))
        y = self.dropout2(self.relu(self.layer14(y)))

        return self.relu(self.output(y))