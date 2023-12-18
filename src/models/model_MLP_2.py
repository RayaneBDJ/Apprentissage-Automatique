import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MLP2Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ASC_TRAIN = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_SM = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_CAR = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_TIME = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_COST = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        
        self.batch1 = torch.nn.BatchNorm1d(240)
        self.batch2 = torch.nn.BatchNorm1d(3)
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

        self.input = torch.nn.Linear(24, 240) # 24 variables en entrée
        self.output = torch.nn.Linear(240, 3) # 3 classes de sortie

    def forward(self, x):
        V1 = self.ASC_TRAIN + self.B_TIME * x[:, 19] + self.B_COST * x[:, 20]
        V2 = self.ASC_SM + self.B_TIME * x[:, 22] + self.B_COST * x[:, 23]
        V3 = self.ASC_CAR + self.B_TIME * x[:, 25] + self.B_COST * x[:, 26]
        # Il est important d'enlever les variables déjà prises en compte dans V :
        # "TRAIN_TT" et "TRAIN_CO", soit [:19, :20]
        # "SM_TT" et "SM_CO", soit [:22, :23]
        # "CAR_TT" et "CAR_CO", soit [:25, :26]
        y = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,24]]
        V = torch.concat((y, V1.unsqueeze(1), V2.unsqueeze(1), V3.unsqueeze(1)), dim=1)
        y = self.batch1(self.dropout(self.relu(self.input(V))))

        return self.batch2(self.dropout(self.output(y)))
    
class MLP2Model8(torch.nn.Module):
    def __init__(self):    
        super().__init__()
        self.ASC_TRAIN = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_SM = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_CAR = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_TIME = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_COST = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)

        self.input = torch.nn.Linear(24, 240) # 24 variables en entrée
        self.layer1 = torch.nn.Linear(240, 200)
        self.layer2 = torch.nn.Linear(200, 140)
        self.layer3 = torch.nn.Linear(140, 100)
        self.layer4 = torch.nn.Linear(100, 60)
        self.layer5 = torch.nn.Linear(60, 45)
        self.layer6 = torch.nn.Linear(45, 30)
        self.output = torch.nn.Linear(30, 3) # 3 classes de sortie

        self.batch1 = torch.nn.BatchNorm1d(240)
        self.batch2 = torch.nn.BatchNorm1d(200)
        self.batch3 = torch.nn.BatchNorm1d(140)
        self.batch4 = torch.nn.BatchNorm1d(100)
        self.batch5 = torch.nn.BatchNorm1d(60)
        self.batch6 = torch.nn.BatchNorm1d(45)
        self.batch7 = torch.nn.BatchNorm1d(30)
        self.batch8 = torch.nn.BatchNorm1d(3)
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        V1 = self.ASC_TRAIN + self.B_TIME * x[:, 19] + self.B_COST * x[:, 20]
        V2 = self.ASC_SM + self.B_TIME * x[:, 22] + self.B_COST * x[:, 23]
        V3 = self.ASC_CAR + self.B_TIME * x[:, 25] + self.B_COST * x[:, 26]
        # Il est important d'enlever les variables déjà prises en compte dans V :
        # "TRAIN_TT" et "TRAIN_CO", soit [:19, :20]
        # "SM_TT" et "SM_CO", soit [:22, :23]
        # "CAR_TT" et "CAR_CO", soit [:25, :26]
        y = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,24]]
        V = torch.concat((y, V1.unsqueeze(1), V2.unsqueeze(1), V3.unsqueeze(1)), dim=1)

        y = self.batch1(self.relu(self.input(V)))
        y = self.batch2(self.relu(self.layer1(y)))
        y = self.batch3(self.relu(self.layer2(y)))
        y = self.batch4(self.relu(self.layer3(y)))
        y = self.batch5(self.relu(self.layer4(y)))
        y = self.batch6(self.relu(self.layer5(y)))
        y = self.batch7(self.relu(self.layer6(y)))

        return self.batch8(self.dropout(self.relu(self.output(y))))
    
class MLP2Model16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ASC_TRAIN = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_SM = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_CAR = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_TIME = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_COST = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)

        self.input = torch.nn.Linear(24, 135) # 24 variables en entrée
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

        self.batch1 = torch.nn.BatchNorm1d(135)
        self.batch2 = torch.nn.BatchNorm1d(70)
        self.batch3 = torch.nn.BatchNorm1d(60)
        self.batch4 = torch.nn.BatchNorm1d(50)
        self.batch5 = torch.nn.BatchNorm1d(45)
        self.batch6 = torch.nn.BatchNorm1d(40)
        self.batch7 = torch.nn.BatchNorm1d(60)
        self.batch8 = torch.nn.BatchNorm1d(55)
        self.batch9 = torch.nn.BatchNorm1d(50)
        self.batch10 = torch.nn.BatchNorm1d(45)
        self.batch11 = torch.nn.BatchNorm1d(40)
        self.batch12 = torch.nn.BatchNorm1d(30)
        self.batch13 = torch.nn.BatchNorm1d(25)
        self.batch14 = torch.nn.BatchNorm1d(20)
        self.batch15 = torch.nn.BatchNorm1d(10)
        self.batch16 = torch.nn.BatchNorm1d(3) 
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        V1 = self.ASC_TRAIN + self.B_TIME * x[:, 19] + self.B_COST * x[:, 20]
        V2 = self.ASC_SM + self.B_TIME * x[:, 22] + self.B_COST * x[:, 23]
        V3 = self.ASC_CAR + self.B_TIME * x[:, 25] + self.B_COST * x[:, 26]
        
        y = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,24]]
        V = torch.cat((y, V1.unsqueeze(1), V2.unsqueeze(1), V3.unsqueeze(1)), dim=1)

        y = self.batch1(self.relu(self.input(V)))
        y = self.batch2(self.relu(self.layer1(y)))
        y = self.batch3(self.relu(self.layer2(y)))
        y = self.batch4(self.relu(self.layer3(y)))
        y = self.batch5(self.relu(self.layer4(y)))
        y = self.batch6(self.relu(self.layer5(y)))
        y = self.batch7(self.relu(self.layer6(y)))
        y = self.batch8(self.relu(self.layer7(y)))
        y = self.batch9(self.relu(self.layer8(y)))
        y = self.batch10(self.relu(self.layer9(y)))
        y = self.batch11(self.relu(self.layer10(y)))
        y = self.batch12(self.relu(self.layer11(y)))
        y = self.batch13(self.relu(self.layer12(y)))
        y = self.batch14(self.relu(self.layer13(y)))
        y = self.batch15(self.relu(self.layer14(y)))
        
        return self.batch16(self.dropout(self.relu(self.output(y))))
