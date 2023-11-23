import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_dataloader():

    data_transforms= transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]),
                    ])
   
    dataset = torchvision.datasets.ImageFolder("/workspace/datasets/ImageNet/val", transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1)

    return dataloader
