import torch
from src.models.model import MyNeuralNet
from tests import _PATH_DATA
def test_output():
    data = torch.empty((1,784))
    model = MyNeuralNet()
    output = model(data)
    ps = torch.exp(output)
    assert ps.shape == torch.Size([1,10])