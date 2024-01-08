import torch
from torchvision import transforms
import glob
import os

def process():
    imgs_path = "data/raw/"
    image = glob.glob(imgs_path + "*train_image*")
    save_path = "data/processed/"
    data = []
    labels = []
    
    # Processed train files
    for image_path in image:
        root,tail = os.path.split(image_path)
        label_path = os.path.join(root, tail.split("_")[0]+"_target_"+tail.split("_")[-1])

        loaded_data = torch.load(image_path)
        loaded_labels = torch.load(label_path)
            
        assert len(loaded_data) == len(loaded_labels)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
            # Add other transformations as needed
        ])

        if not isinstance(loaded_data, torch.Tensor):
            loaded_data = transform(loaded_data)
            
        print(len(loaded_data))
        data.extend(loaded_data)
        labels.extend(loaded_labels)
    torch.save(data, save_path +'train_image.pt') 
    torch.save(labels, save_path +'train_label.pt')   
    
    # Processed test files
    image_path = os.path.join(imgs_path, "test_images.pt")
    label_path = os.path.join(imgs_path, "test_target.pt")
    
    test_data = torch.load(image_path)
    test_labels = torch.load(label_path)
            
    assert len(test_data) == len(test_labels)

    if not isinstance(test_data, torch.Tensor):
        test_data = transform(test_data)
    torch.save(data, save_path +'test_image.pt') 
    torch.save(labels, save_path +'test_label.pt') 

if __name__ == '__main__':
    # Get the data and process it
    process()
    pass