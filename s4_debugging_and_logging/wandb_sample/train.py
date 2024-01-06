import wandb
import random
import sys
print(sys.executable)

cfg_default = dict (
        learning_rate = 0.02,
        architecture ="CNN",
        dataset = "CIFAR-100",
        epochs = 10,
    )
# start a new wandb run to track this script
wandb.init(project="exercise_wandb",config= cfg_default)
config = wandb.config

# simulate training
epochs = config.epochs
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

