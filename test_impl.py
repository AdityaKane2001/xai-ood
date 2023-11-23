from probe import Prober
from utils import find_imagenet_label, plot_mask_and_im
from PIL import Image
import torch


p = Prober("deit")
lbl = find_imagenet_label("n01498041")

if lbl is not None:
    lbl = lbl["idx"]

im = Image.open("/workspace/akane/xai-ood/imagenet-a/n01498041/0.000116_digital clock _ digital clock_0.865662.jpg")

g, a = p.compute_explanations(im, torch.tensor([int(lbl)]), attack=True)
# print(g)
# print(a)

import matplotlib.pyplot as plt
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
ax.imshow(a)
fig.savefig('a.png')   # save the figure to file
plt.close(fig)  

plot_mask_and_im(im, a, savepath="here.png")