from typing import Any
import torch
import torchattacks
# from dataset import get_dataloader
from model import get_model
from utils import save_images

DEVICE = "cuda"
# def PGD(model, dataloader, save_path):
#     print("PGD")
#     atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
#     atk.set_normalization_used(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])

#     for batch in dataloader:
#         images, labels = batch
#         adv_images = atk(images, labels)
#         save_images(images, adv_images, save_path)

#         break ### running just for one batch for now

class Attacker:
    def __init__(self, model, pgd_eps=8/255, pgd_alpha=2/255, pgd_steps=4) -> None:
        self.atk = torchattacks.PGD(model, eps=pgd_eps, alpha=pgd_alpha, steps=pgd_steps)
        self.atk.set_normalization_used(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
        self.device = 'cuda'
    def __call__(self, imgs, labels, savepath=None):
        
        single_img = False        
        if len(imgs.shape) < 4:
            single_img = True
            imgs = imgs.unsqueeze(0)
        
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        adv_images = self.atk(imgs, labels)
        
        if savepath is not None:
            save_images(imgs, adv_images, savepath)
        
        if single_img:
            adv_images = adv_images.squeeze()
        
        return adv_images
        

# if __name__ == '__main__':
#     model = get_model("vitb16")
#     dataloader = get_dataloader()
#     PGD(model, dataloader, "./examples")