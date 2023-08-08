import torch
from torch.nn import MSELoss
from torch.optim import Adam

from PIL import Image

import torchvision.transforms.functional as TF

import sys
sys.path.append("/home/kdhsimplepro/kdhsimplepro/AI/Stylers/UNet")

from models import UNet


unet = UNet().to("cuda")

unet.load_state_dict(torch.load("/home/kdhsimplepro/kdhsimplepro/AI/Stylers/UNet/best_params.pth"))
unet.eval()

@torch.no_grad()
def pred_unet(x):

    mask = unet(x)
    return mask

original_path = "./V.png"
transformed_path = "./v_test.png"

original_img = TF.to_tensor(Image.open(original_path).resize((256, 256)).convert("RGB")).to("cuda").unsqueeze(0)
transformed_img = TF.to_tensor(Image.open(transformed_path).resize((256, 256)).convert("RGB")).to("cuda").unsqueeze(0)
transformed_img.requires_grad = True

original_mask = pred_unet(original_img)
transformed_mask = pred_unet(transformed_img).detach()

optim = Adam([transformed_img], lr=0.002)

label = original_img * (1 - original_mask)

criterion = MSELoss()

imgs = []


interpolated_mask = (original_mask + transformed_mask).clamp_(0, 1).cuda().squeeze(0)

def interpolate_img(original_img, transformed_img, interpolated_mask):
    interpolated_img = original_img * (1-interpolated_mask) + interpolated_mask * transformed_img
    return interpolated_img

interpolated_img = interpolate_img(original_img, transformed_img, interpolated_mask)


def make_gif(img_list, save_path, fps=50):
    img, *imgs = [TF.to_pil_image(im) for im in img_list]
    img.save(fp=save_path, format="GIF", append_images=imgs, save_all=True, duration=fps, loop=1)

coordinates = torch.zeros((256, 256, 2))
for i in range(256):
    for j in range(256):
        coordinates[i, j, 0] = i
        coordinates[i, j, 1] = j

def get_gaussian_dist(x, y, size=256):
    matrix = torch.zeros((size, size))

    dist = torch.distributions.normal.Normal(torch.tensor([0]), torch.tensor([size])/2)
    distance = torch.sum(torch.abs(coordinates - torch.Tensor([y, x])), dim=2)
    matrix = torch.exp(dist.log_prob(distance))
    matrix /= torch.sum(matrix)
    
    return matrix

gaussian_matrix = torch.zeros((256, 256))

for i in range(256):
    for j in range(256):
        if interpolated_mask[0, i, j] >= 0.7:
            gaussian_matrix += get_gaussian_dist(i, j, 256)

gaussian_matrix /= torch.sum(gaussian_matrix)
gaussian_matrix = gaussian_matrix.to("cuda")
import matplotlib.pyplot as plt
plt.imshow(gaussian_matrix.cpu().detach().permute(1, 0).numpy().reshape(256, 256, 1), cmap="gray")
plt.show()

interpolated_img = interpolated_img.squeeze(0)

alpha = 0.5
for k in range(100):

    with torch.no_grad():
        imgs.append(transformed_img.clamp_(0, 1).detach().cpu().squeeze(0).clamp_(0, 1))

    loss = criterion(transformed_img, interpolated_img.unsqueeze(0))

    optim.zero_grad()
    loss.backward(retain_graph=True)
    with torch.no_grad():
        mean_grad = torch.mean(transformed_img.grad)
        expectation_grad = torch.sum(transformed_img.grad * gaussian_matrix.reshape(1, 1, 256, 256).repeat(1, 3, 1, 1))
        print(mean_grad,expectation_grad)
        for i in range(256):
            for j in range(256):
                if interpolated_mask[:, i, j] >= 0.7:
                    transformed_img.grad[:, :, i, j] = expectation_grad
                
                else:
                    transformed_img.grad[:, :, i, j] = (alpha * transformed_img.grad[:, :, i, j] + (1-alpha)*mean_grad)

    optim.step()

    print(k, round(loss.item(), 4))

make_gif(imgs, save_path="./corrected_face_v.gif")