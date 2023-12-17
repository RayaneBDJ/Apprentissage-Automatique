import torch

from models.residual_components import OutputLayer, ResidualBlock

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ResLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ASC_TRAIN = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_SM = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_CAR = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_TIME = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_COST = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.residual_block = ResidualBlock(24, 24, n_layers=6)
        # self.residual_block1 = ResidualBlock(24, 24)
        # self.residual_block2 = ResidualBlock(24, 24)
        self.output = OutputLayer(24, 3)

    def forward(self, x):
        V1 = self.ASC_TRAIN + self.B_TIME * x[:, 19] + self.B_COST * x[:, 20]
        V2 = self.ASC_SM + self.B_TIME * x[:, 22] + self.B_COST * x[:, 23]
        V3 = self.ASC_CAR + self.B_TIME * x[:, 25] + self.B_COST * x[:, 26]
        y = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,24]] # enlève col TRAIN [:19, :20], SM [:22, :23], CAR [:25, :26]
        V = torch.concat((y, V1.unsqueeze(1), V2.unsqueeze(1), V3.unsqueeze(1)), dim=1)

        U = self.residual_block(V)

        return self.output(torch.softmax(V, dim=1))

        # U = self.residual_block1(V)
        # W = self.residual_block2(U)
        # return self.output(torch.softmax(W, dim=1))
