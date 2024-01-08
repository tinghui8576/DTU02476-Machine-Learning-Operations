import torch
import os
from src.models.model import MyNeuralNet
from src.train_model import TrainDataset
from torch.utils.data import DataLoader
from tests import _PATH_DATA
def test_output():
    trainset = TrainDataset()
    data = trainset[1]["data"]
    data= data.view(1,784)
    model = MyNeuralNet()
    assert data.shape == torch.Size([1,784])
    output = model(data)
    ps = torch.exp(output)
    assert ps.shape == torch.Size([1,10])