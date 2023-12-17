from functools import reduce
import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ResidualLayer(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super(ResidualLayer, self).__init__()
        self.weights = torch.nn.Parameter(torch.rand(n_features_in, n_features_out), requires_grad=True)
    
    def forward(self, V):
        return torch.nn.functional.softplus(torch.matmul(V, self.weights))


class ResidualBlock(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out, n_layers=16):
        super(ResidualBlock, self).__init__()

        self.layers = torch.nn.ModuleList([
            ResidualLayer(n_features_in, n_features_out) for _ in range(n_layers)
        ])

        self.cumulate = lambda resultats: reduce(lambda x, y: x + y, resultats) if len(resultats) > 0 else 0
    
    def forward(self, V):

        layer_results = []
        out = V
        for layer in self.layers:
            layer_results.append(layer(out))
            out = out - self.cumulate(layer_results)

        return V - self.cumulate(layer_results)


class OutputLayer(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super(OutputLayer, self).__init__()
        self.weights = torch.nn.Parameter(torch.rand(n_features_in, n_features_out), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand(n_features_out,), requires_grad=True)

    def forward(self, V):
        return torch.relu(torch.matmul(V, self.weights) + self.bias)


class SwissMetroResLogit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ASC_TRAIN = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_SM = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.ASC_CAR = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_TIME = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.B_COST = torch.nn.Parameter(torch.rand(1, dtype=torch.float, device=DEVICE), requires_grad=True)
        self.residual_block1 = ResidualBlock(24, 24)
        self.output = OutputLayer(24, 3)

    def forward(self, x):
        V1 = self.ASC_TRAIN + self.B_TIME * x[:, 19] + self.B_COST * x[:, 20]
        V2 = self.ASC_SM + self.B_TIME * x[:, 22] + self.B_COST * x[:, 23]
        V3 = self.ASC_CAR + self.B_TIME * x[:, 25] + self.B_COST * x[:, 26]
        y = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,24]] # enlève col TRAIN [:19, :20], SM [:22, :23], CAR [:25, :26]
        V = torch.concat((y, V1.unsqueeze(1), V2.unsqueeze(1), V3.unsqueeze(1)), dim=1)

        U1 = self.residual_block1(V)
        expU = torch.exp(U1)
        return self.output(expU / torch.sum(expU, dim=1).unsqueeze(1))
