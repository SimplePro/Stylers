from core.checkpoint import CheckpointIO

from torch import nn
from core.model import Generator, MappingNetwork, StyleEncoder, Discriminator, FAN

import copy

from munch import Munch

import torch
import torchvision.transforms as TF

from PIL import Image

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

y_ref = 0 # forehead0: 0, forehead1: 1
# y_ref = torch.tensor([y_ref], dtype=torch.bool).to(device)

x_ref = TF.ToTensor()(Image.open("")).unsqueeze(0).to(device)

s_ref = nets_ema.style_encoder(x_ref, y_ref)

x_src = TF.ToTensor()(Image.open("")).unsqueeze(0).to(device)

masks = nets_ema.fan.get_heatmap(x_src)
fake = nets_ema.generator(x_src, s_ref, masks=masks)

print(fake.shape)