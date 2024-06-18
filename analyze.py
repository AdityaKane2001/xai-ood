import numpy as np
import matplotlib.pyplot as plt

from utils import get_spread

from probe import Prober
# from utils import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataset import ImageNetValDataset, ImageNetADataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(43)
# torch.seed(42)

## Add folder path below
data_folder = "./imagenet-val"
# data_folder = "/satassdscratch/amete7/xai-ood/imagenet-a"

## Since val labels are in one txt for me
labels_file = "./imagenet_val_labels.txt"

transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]),
                ])
custom_dataset = ImageNetValDataset(data_folder, labels_file, transform)

def load_and_print(filename):
    print(filename.split("/")[-1])
    arr = np.load(filename)
    # print(arr)
    print(arr.mean())
    print("=" * 20)
    return arr


def plot_and_save_attn(attn, atk_attn, pil_img, path="./scratch.jpeg"):
    fig, ax = plt.subplots( nrows=3, ncols=1 )  # create figure & 1 axis
    ax[0].imshow(attn)
    ax[1].imshow(atk_attn)
    ax[2].imshow(pil_img)
    ax[0].set_title('Grad Attention')
    ax[1].set_title('Attack GA')
    ax[2].set_title('Image')
    fig.suptitle(f"Orig score: {id_scores[idx]} Atk score: {id_atk_scores[idx]}")
    fig.savefig(path)
    plt.close()
    
def plot_and_save(idx, path="./scratch.jpeg"):
    print(f"ID score: {id_scores[idx]}")
    print(f"ID attack score: {id_atk_scores[idx]}")
    attn_ = attn[idx]
    atk_attn_ = atk_attn[idx]
    _,_,img = custom_dataset[idx]
    plot_and_save_attn(attn_, atk_attn_, img, path=path)
    
# id = load_and_print("/workspace/akane/xai-ood/id_099.npy")
# id_atk = load_and_print("/workspace/akane/xai-ood/ood_imgA_099.npy")
# # id_atk = load_and_print("/workspace/akane/xai-ood/ood_imgO_099.npy")
# id = load_and_print("/workspace/akane/xai-ood/id_scores.npy")
# id_atk = load_and_print("/workspace/akane/xai-ood/id_attack_scores.npy")
attn = np.load("/workspace/akane/xai-ood/id_attn.npy")
atk_attn = np.load("/workspace/akane/xai-ood/id_attn_attack.npy")
img0_attn = np.load("/workspace/akane/xai-ood/ood_imgO_full_attn.npy")


id_scores = list()
id_atk_scores = list()
img0_scores = list()

print(attn.shape)

for map, amap in zip(attn, atk_attn):
    id_scores.append(get_spread(map))
    id_atk_scores.append(get_spread(amap))

for img0map in img0_attn:
    img0_scores.append(get_spread(img0map))

# id_full = load_and_print("/workspace/akane/xai-ood/id_full.npy")

# print(np.mean(id_full))
# print(id_scores[:10])
# print(np.mean(id_atk_scores))

# load_and_print("/workspace/akane/xai-ood/id_099.npy")
# load_and_print("/workspace/akane/xai-ood/id_099_attack.npy")
# load_and_print("//workspace/akane/xai-ood/ood_imgA_099.npy")
# load_and_print("//workspace/akane/xai-ood/ood_imgO_099.npy")

# for i in range(100):
#     idx = np.random.randint(0, 9000)
#     plot_and_save(idx, path=f"./scratch_ims/{i}")

train_id_scores = np.array(id_scores)[:int(len(id_scores) * 0.9)]
train_id_atk_scores  = np.array(id_atk_scores)[:int(len(id_scores) * 0.9)]

test_id_scores = np.array(id_scores)[int(len(id_scores) * 0.9):]
test_id_atk_scores = np.array(id_atk_scores)[int(len(id_atk_scores) * 0.9):]

img0_full = np.array(img0_scores)
# img0_full = load_and_print("/workspace/akane/xai-ood/ood_imgO_full.npy")

good_id_scores = train_id_scores[train_id_scores < train_id_atk_scores]
good_id_atk_scores = train_id_atk_scores[train_id_scores < train_id_scores]


id_counts, _, _ = plt.hist(good_id_scores,  bins=np.linspace(start=0, stop=40, num=80), histtype="step", color="red", weights=np.ones(len(good_id_scores)) / len(good_id_scores), label="ID") #weights=np.ones(len(good_id_scores)) / len(good_id_scores),
plt.show()
img0_counts, _, _ = plt.hist(img0_full, weights=np.ones(len(img0_full)) / len(img0_full),bins=np.linspace(start=0, stop=40, num=80), histtype="step", color="green", label="Unknown OOD")
plt.show()
id_atk_counts, _, _ = plt.hist(good_id_atk_scores,  bins=np.linspace(start=0, stop=40, num=80), histtype="step", color="blue", weights=np.ones(len(good_id_atk_scores)) / len(good_id_atk_scores), label="Known OOD") #weights=np.ones(len(good_id_atk_scores)) / len(good_id_atk_scores),
plt.show()
# print(id_counts)

from scipy.stats import norm
print(np.mean(good_id_scores))
print(np.std(good_id_scores))
id_mu, id_std = norm.fit(good_id_scores)
print(id_mu, id_std)
# plt.axvline(x=id_mu)
# plt.axvline(x=id_mu + id_std)
plt.legend()
plt.title("Comparing ID, Known OOD and Unknown OOD")
plt.savefig("hist_correct_chosen.jpeg")

test_data = np.concatenate([test_id_scores, img0_full])
test_labels = np.concatenate([np.zeros(len(test_id_scores)), np.ones(len(img0_full))])

test_preds = (test_data > id_mu)
print("Acc:", np.sum((test_preds == test_labels).astype(float))/len(test_labels))


print(np.sum((img0_scores > np.mean(good_id_scores).astype(float))))

# argsorted = np.argsort(id_full)
# print(id_full[argsorted[int(0.6 * len(id_full))]])

# plt.hist(id_full, bins=np.linspace(start=0, stop=10, num=50), histtype="step", weights=np.ones(len(id_full)) / len(id_full))
# plt.show()
# plt.hist(id_atk, bins=np.linspace(start=0, stop=10, num=50), histtype="step", weights=np.ones(len(id_atk)) / len(id_atk))
# plt.show()
# plt.savefig("ID_ATK.jpeg")

# id = load_and_print("/workspace/akane/xai-ood/ood_imgO_095.npy")
# id_atk = load_and_print("/workspace/akane/xai-ood/ood_imgO_095_attack.npy")


# ood_O = load_and_print("/workspace/akane/xai-ood/ood_imgO_scores.npy")
# ood_O_atk = load_and_print("/workspace/akane/xai-ood/ood_imgO_attack_scores.npy")


# ood_A = load_and_print("/workspace/akane/xai-ood/ood_imgA_scores.npy")
# ood_A_atk = load_and_print("/workspace/akane/xai-ood/ood_imgA_attack_scores.npy"