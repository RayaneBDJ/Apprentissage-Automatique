from functools import reduce
import torch

__all__ = ["ResidualBlock", "OutputLayer"]

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
        h = V
        for layer in self.layers:
            layer_result = layer(h)
            layer_results.append(layer_result)
            h = h - layer_result

        return V - self.cumulate(layer_results)


class OutputLayer(torch.nn.Module):
    def __init__(self, n_features_in, n_features_out):
        super(OutputLayer, self).__init__()
        self.weights = torch.nn.Parameter(torch.rand(n_features_in, n_features_out), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.rand(n_features_out,), requires_grad=True)

    def forward(self, V):
        return torch.nn.functional.relu(torch.matmul(V, self.weights) + self.bias)