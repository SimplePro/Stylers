from core.checkpoint import CheckpointIO

from torch import nn
from torchvision.utils import make_grid

from core.model import Generator, MappingNetwork, StyleEncoder, Discriminator, FAN

import copy

from munch import Munch

import torch
import torchvision.transforms as TF

from PIL import Image

from preprocessing import Preprocessing

img_size = 256
style_dim = 64
latent_dim = 16
w_hpf = 1e-8
num_domains = 2

generator = nn.DataParallel(Generator(img_size, style_dim, w_hpf=w_hpf))
mapping_network = nn.DataParallel(MappingNetwork(latent_dim, style_dim, num_domains))
style_encoder = nn.DataParallel(StyleEncoder(img_size, style_dim, num_domains))
discriminator = nn.DataParallel(Discriminator(img_size, num_domains))
generator_ema = copy.deepcopy(generator)
mapping_network_ema = copy.deepcopy(mapping_network)
style_encoder_ema = copy.deepcopy(style_encoder)

nets_ema = Munch(generator=generator_ema, mapping_network=mapping_network_ema, style_encoder=style_encoder_ema)

fan = nn.DataParallel(FAN(fname_pretrained="expr/checkpoints/wing.ckpt").eval())
fan.get_heatmap = fan.module.get_heatmap
nets_ema.fan = fan


ckptios = CheckpointIO('expr2/checkpoints/100000_nets_ema.ckpt', data_parallel=True, **nets_ema)
ckptios.load(100000)

device = "cuda" if torch.cuda.is_available() else "cpu"

y_ref = 1 # forehead0: 0, forehead1: 1

src_path = "../data/augmentation_data/forehead/valid/0/80000.jpg"
ref_path = "../data/augmentation_data/forehead/valid/1/80005.jpg"

transform = TF.Compose([
        TF.Resize((256, 256)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

preprocessing = Preprocessing(device="cuda")
preprocessed_ref = preprocessing.preprocess_(src_path=src_path, ref_path=ref_path)

import matplotlib.pyplot as plt

plt.imshow(preprocessed_ref)
plt.show()

x_ref = transform(Image.open(ref_path)).unsqueeze(0).to(device)
# x_ref = TF.ToTensor()(Image.fromarray(preprocessed_ref)).unsqueeze(0).to(device)

s_ref = nets_ema.style_encoder(x_ref, y_ref)

x_src = transform(Image.open(src_path)).unsqueeze(0).to(device)

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

masks = nets_ema.fan.get_heatmap(x_src)
fake = denormalize(nets_ema.generator(x_src, s_ref, masks=masks))
fake_img = TF.ToPILImage()(fake.cpu().detach().squeeze(0))

print(fake)
print(fake.shape)
fake_img.save("./test1.png")