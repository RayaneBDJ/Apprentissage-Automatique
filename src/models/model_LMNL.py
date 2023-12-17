import torch

from models.residual_components import OutputLayer, ResidualBlock

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LMNLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ASC_TRAIN = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_SM = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_CAR = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_TIME = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_COST = torch.nn.Parameter(torch.randn(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.residual_block = ResidualBlock(27, 27)
        self.output = OutputLayer(24, 3)

    def forward(self, x):
        U = self.residual_block(x)

        V1 = self.ASC_TRAIN + self.B_TIME * U[:, 19] + self.B_COST * U[:, 20]
        V2 = self.ASC_SM + self.B_TIME * U[:, 22] + self.B_COST * U[:, 23]
        V3 = self.ASC_CAR + self.B_TIME * U[:, 25] + self.B_COST * U[:, 26]
        # Il est important d'enlever les variables déjà prises en compte dans V :
        # "TRAIN_TT" et "TRAIN_CO", soit [:19, :20]
        # "SM_TT" et "SM_CO", soit [:22, :23]
        # "CAR_TT" et "CAR_CO", soit [:25, :26]
        y = U[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,24]]
        V = torch.concat((y, V1.unsqueeze(1), V2.unsqueeze(1), V3.unsqueeze(1)), dim=1)

        return self.output(torch.softmax(V, dim=1))