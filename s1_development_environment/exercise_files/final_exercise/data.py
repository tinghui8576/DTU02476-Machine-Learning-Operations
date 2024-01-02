import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self):
        self.imgs_path = "/Users/tinghui/workspace/python/dtu_mlops/data/corruptmnist/"
        
        image = glob.glob(self.imgs_path + "*train_image*")
        # label = glob.glob(self.imgs_path + "*train_target*")
        self.data = []
        self.labels = []
        for image_path in image:
            
            root,tail = os.path.split(image_path)
            label_path = os.path.join(root, tail.split("_")[0]+"_target_"+tail.split("_")[-1])

            loaded_data = torch.load(image_path)
            loaded_labels = torch.load(label_path)
            
            assert len(loaded_data) == len(loaded_labels)
            
            self.data.extend(loaded_data)
            self.labels.extend(loaded_labels)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Add other transformations as needed
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }

        # Apply transformations if needed
        if self.transform:
            if not isinstance(sample['data'], torch.Tensor):
                sample['data'] = self.transform(sample['data'])

        return sample

class TestDataset(Dataset):
    def __init__(self):
        self.imgs_path = "/Users/tinghui/workspace/python/dtu_mlops/data/corruptmnist/"
        
        image_path = os.path.join(self.imgs_path, "test_images.pt")
        label_path = os.path.join(self.imgs_path, "test_target.pt")
        
        self.data =  torch.load(image_path)
        self.labels =  torch.load(label_path)

        assert len(self.data) == len(self.labels)

        # You may need to preprocess the data or apply transformations here
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Add other transformations as needed
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }

        # Apply transformations if needed
        if self.transform:
            if not isinstance(sample['data'], torch.Tensor):
                sample['data'] = self.transform(sample['data'])

        return sample


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    trainset = TrainDataset()
    testset= TestDataset()
    batch_size = 64  # Adjust this based on your requirements
    shuffle = True    # You may want to shuffle the data during training
    train = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
    test = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train, test
