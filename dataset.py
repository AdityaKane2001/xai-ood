import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os
from PIL import Image

class ImageNetValDataset(Dataset):
    def __init__(self, data_folder, labels_file, transform=None):
        self.data_folder = data_folder
        self.labels_file = labels_file
        self.transform = transform

        # Read labels from the text file
        with open(labels_file, 'r') as file:
            self.labels = [int(label.strip()) for label in file.readlines()]

        # Ensure the number of labels matches the number of images
        self.num_images = len(os.listdir(data_folder))
        assert len(self.labels) == self.num_images, "Number of labels doesn't match the number of images."

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Construct the image file path
        img_name = f"ILSVRC2012_val_{idx + 1:08d}.JPEG"  # Assuming images are named 1.jpeg, 2.jpeg, ...
        img_path = os.path.join(self.data_folder, img_name)

        # Load the image
        img_original = Image.open(img_path).convert("RGB")

        # Load the label
        label = self.labels[idx]

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img_original)

        return img, label, img_original

class ImageNetADataset(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        target = path.split("/")[-2]
        original_image = Image.open(path).convert('RGB')

        if self.transform is not None:
            transformed_image = self.transform(original_image)
        else:
            transformed_image = original_image

        if self.target_transform is not None:
            # print(target)
            target = self.target_transform(target)["idx"]

        return transformed_image, target, original_image


def get_dataset(dataset_path,data_transforms):
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=data_transforms)
    return dataset

if __name__ == "__main__":
    data_folder = "/satassdscratch/amete7/xai-ood/imagenet_val"  # Replace with the actual path
    labels_file = "/satassdscratch/amete7/xai-ood/imagenet_val_labels.txt"    # Replace with the actual path
    transform = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]),
                    ])
    custom_dataset = ImageNetValDataset(data_folder, labels_file, transform)

    # Example: Print the first image and label
    sample_img, sample_label = custom_dataset[17]
    print(f"Label: {sample_label}")
    print(sample_img.shape)