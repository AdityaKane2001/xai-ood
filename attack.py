import torch
import torchattacks
from dataset import get_dataloader
from model import get_model
from utils import save_images


def PGD(model, dataloader, save_path):
    print("PGD")
    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
    atk.set_normalization_used(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])

    for batch in dataloader:
        images, labels = batch
        adv_images = atk(images, labels)
        save_images(images, adv_images, save_path)

        break ### running just for one batch for now


if __name__ == '__main__':
    model = get_model("vitb16")
    dataloader = get_dataloader()
    PGD(model, dataloader, "./examples")


    
    
