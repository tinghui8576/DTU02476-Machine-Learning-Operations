import click
import torch
import os
from torch.utils.data import Dataset,DataLoader
from models.model import MyNeuralNet
from torch import nn
from torch import optim
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self):
        self.imgs_path = "data/processed/"
        
        
        image_path = os.path.join(self.imgs_path, "train_image.pt")
        label_path = os.path.join(self.imgs_path, "train_label.pt")

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

class TestDataset(Dataset):
    def __init__(self):
        self.imgs_path = "data/processed/"
        
        image_path = os.path.join(self.imgs_path, "test_image.pt")
        label_path = os.path.join(self.imgs_path, "test_label.pt")
        
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


def dataloader():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    trainset = TrainDataset()
    testset= TestDataset()
    batch_size = 64  # Adjust this based on your requirements
    shuffle = True    # You may want to shuffle the data during training
    train = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    test = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train, test

def plot_fig(train_losses, test_losses):
    


    epochs = range(1, len(train_losses) + 1)

    # Plotting train and test losses
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')

    plt.title('Train and Test Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('reports/figures/training_curve.png')



@click.group()
def cli():
    """Command line interface."""
    pass

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

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyNeuralNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 30
    steps = 0
    running_loss = 0

    train_set, test_set = dataloader()


    train_losses, test_losses = [], []

    for e in range(epochs):
        for batch in train_set:
            steps += 1
            # load data and labels in the batch
            data = batch['data']
            labels = batch['label']

            # Training
            optimizer.zero_grad()
            log_ps = model(data)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        else:
            model.eval()

            # Turn off gradients for validation, will speed up inference
            with torch.no_grad():
                test_loss, accuracy = validation(model, test_set, criterion)

            train_loss = running_loss/len(train_set)
            train_losses.append(train_loss)
            test_loss = test_loss / len(test_set)
            test_losses.append(test_loss)
            print(
                "Epoch: {}/{} ".format(e + 1, epochs),
                "Training Loss: {:.3f} ".format(train_loss),
                "Test Loss: {:.3f} ".format(test_loss),
                "Test Accuracy: {:.3f}".format(accuracy / len(test_set)),
                )
            running_loss = 0
            model.train()
    plot_fig(train_losses, test_losses)
    torch.save(model.state_dict(), 'models/trained_model.pt')



cli.add_command(train)


if __name__ == "__main__":
    cli()
