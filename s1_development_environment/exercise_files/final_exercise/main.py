import click
import torch
from model import MyAwesomeModel
from torch import nn
from torch import optim

from data import mnist


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
    model = MyAwesomeModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 30
    steps = 0
    running_loss = 0

    train_set, test_set = mnist()


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
    torch.save(model.state_dict(), 'trained_model.pt')

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)  
     
    _, test_set = mnist()
    criterion = nn.CrossEntropyLoss()
    _, accuracy = validation(model, test_set,criterion)
    print("Accuracy: {:.3f}".format(accuracy / len(test_set)))


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
