import torch
import os
from src.models.model import MyNeuralNet
from torch import nn
import src.train_model as training
from tests import _PATH_DATA
def test_dataloader():
    train,test = training.dataloader()
    assert len(train) > 0 and len(test) >0
def test_train():
    # trainset = training.TrainDataset()
    # data = trainset[1]["data"]
    # labels = trainset[1]["label"]
    # data= data.view(1,784)
    # model = MyNeuralNet()
    # criterion = nn.CrossEntropyLoss()
    # output = model(data)
    # ps = torch.exp(output)
    # loss = criterion(output , labels)

    model = MyNeuralNet()

    criterion = nn.CrossEntropyLoss()

    train_set, test_set = training.dataloader()

    for batch in train_set:
        # load data and labels in the batch
        data = batch['data']
        labels = batch['label']

        # Training
        log_ps = model(data)
        loss = criterion(log_ps, labels)
        loss.backward()
        assert isinstance(loss.item(), float)
        assert not torch.any(torch.isnan(loss)).item()
