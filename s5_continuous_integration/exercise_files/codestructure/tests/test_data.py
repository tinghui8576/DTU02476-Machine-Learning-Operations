import os
import torch
import pytest
from tests import _PATH_DATA
# To skip your data tests if the corresponding data files does not exist
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    train = torch.load(os.path.join(_PATH_DATA, "train_image.pt"))
    test = torch.load(os.path.join(_PATH_DATA, "test_image.pt"))
    assert len(train) == len(test)
    for data in train:
        assert data.shape == torch.Size([28,28]) or data.shape == torch.Size([784])
