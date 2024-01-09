import torch
import torch.nn.functional as F

class MyNeuralNet(torch.nn.Module):
    """ Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """
    def __init__(self) -> None:
        super().__init__()
        self.input = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 10)

        #Dropout with 0.2 probability
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        # if x.ndim != 4:
        #     raise ValueError('Expected input to a 4D tensor')
        # if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
        #     raise ValueError('Expected each sample to have shape [1, 28, 28]')
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        x = F.log_softmax(self.fc4(x), dim=1)

        return x
