import os
import torch

from tests import _PATH_DATA
def test_data():
    train = torch.load(os.path.join(_PATH_DATA, "train_image.pt"))
    test = torch.load(os.path.join(_PATH_DATA, "test_image.pt"))
    assert len(train) == len(test)
    for data in train:
        assert data.shape == torch.Size([28,28]) or data.shape == torch.Size([784])
