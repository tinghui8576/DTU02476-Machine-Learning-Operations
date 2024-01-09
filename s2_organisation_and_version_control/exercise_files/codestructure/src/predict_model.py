import torch
import os

from torch.utils.data import Dataset,DataLoader
from models.model import MyNeuralNet
import sys

class TestDataset(Dataset):
    def __init__(self, file):

        image_path = os.path.join(file, "test_image.pt")
        label_path = os.path.join(file, "test_label.pt")

        self.data = torch.load(image_path)
        self.labels = torch.load(label_path)

        assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
        return sample
def validation(model, testloader, criterion):
    """Validate the model on the testdata by calculating the sum of mean loss and mean accuracy for each test batch.

    Arguments:
        model: torch network
        testloader: torch.utils.data.DataLoader, dataloader of test set
        criterion: loss function
    """
    accuracy = 0
    test_loss = 0
    for batch in testloader:

        # load data and labels in the batch
        data = batch['data']
        labels = batch['label']

        output = model.forward(data)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = labels.data == ps.max(1)[1]
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def loadfile(model_pt, test):
    model = MyNeuralNet()
    state_dict = torch.load(model_pt)
    model.load_state_dict(state_dict)

    testset= TestDataset(test)
    test_set =DataLoader(testset, batch_size=64, shuffle=False)

    return model, test_set


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    predict = []
    for batch in dataloader:

        # load data and labels in the batch
        data = batch['data']

        output = model.forward(data)

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        predict.append(ps.max(1)[1])

    return predict

if __name__ == "__main__":
    model, path = loadfile(sys.argv[1], str(sys.argv[2]))
    print(predict(model, path))
