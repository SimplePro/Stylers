import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import vgg19, VGG19_Weights

from PIL import Image

import torchvision.transforms.functional as TF

import sys
sys.path.append("/home/kdhsimplepro/kdhsimplepro/AI/Stylers/UNet")

from models import UNet

from copy import deepcopy

from tqdm import tqdm

import gc


unet = UNet().to("cuda")

unet.load_state_dict(torch.load("/home/kdhsimplepro/kdhsimplepro/AI/Stylers/UNet/best_params.pth"))
unet.eval()

@torch.no_grad()
def pred_unet(x):

    mask = unet(x)
    return mask

feature_extractor = vgg19(weights=VGG19_Weights.DEFAULT).features.to("cuda")

# original_path = "/home/kdhsimplepro/Pictures/Screenshots/kdh_face.png"
# transformed_path = "./kdh_test.png"
original_path = "./V.png"
transformed_path = "./v_test.png"

original_img = TF.to_tensor(Image.open(original_path).resize((256, 256)).convert("RGB")).to("cuda").unsqueeze(0)
transformed_img = TF.to_tensor(Image.open(transformed_path).resize((256, 256)).convert("RGB")).to("cuda").unsqueeze(0)
transformed_img.requires_grad = True

original_mask = pred_unet(original_img)
transformed_mask = pred_unet(transformed_img).detach()

optim = Adam([transformed_img], lr=0.002)

label = original_img * (1 - original_mask)
label_latent_vector = feature_extractor(label)

criterion = MSELoss()

imgs = []


interpolated_mask = (original_mask + transformed_mask).clamp_(0, 1).cuda().squeeze(0)
# interpolated_mask = (interpolated_mask >= 0.3).type(torch.float)

def interpolate_img(original_img, transformed_img, interpolated_mask):
    interpolated_img = original_img * (1-interpolated_mask) + interpolated_mask * transformed_img
    return interpolated_img

interpolated_img = interpolate_img(original_img, transformed_img, interpolated_mask)


def make_gif(img_list, save_path, fps=50):
    img, *imgs = [TF.to_pil_image(im) for im in img_list]
    img.save(fp=save_path, format="GIF", append_images=imgs, save_all=True, duration=fps, loop=1)

# def make_correlation_matrix(img):
#     # img shape: (3, 256, 256)
#     correlation_matrix = torch.zeros_like(img)
#     correlation_matrix[:, :, 1:] = img[:, :, 1:] / img[:, :, :-1]
#     correlation_matrix[:, 1:, 0] = img[:, 1:, 0] / img[:, :-1, 0]
#     correlation_matrix = torch.nan_to_num(correlation_matrix, nan=0.0)
#     return correlation_matrix

coordinates = torch.zeros((256, 256, 2))
for i in range(256):
    for j in range(256):
        coordinates[i, j, 0] = i
        coordinates[i, j, 1] = j

def get_gaussian_dist(x, y, size=256):
    matrix = torch.zeros((size, size))

    dist = torch.distributions.normal.Normal(torch.tensor([0]), torch.tensor([size])/32)
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

interpolated_img = interpolated_img.squeeze(0)
# corrected_interpolated_img = interpolated_img


# imgs = []
# for y in tqdm(range(256)):
#     for x in range(256):

#         gaussian_matrix = get_gaussian_dist(x, y, size=256).unsqueeze(0).repeat(3, 1, 1)

#         if interpolated_mask[0, y, x] >= 0.3:
#             corrected_interpolated_img[:, y, x] = torch.sum((gaussian_matrix*0.5) * interpolated_img, dim=(1, 2)) + (0.5) * (interpolated_img[:, y, x])
#             # temp = interpolated_img
#             # next_interpolated_img = (1-gaussian_matrix) * temp + (gaussian_matrix) * (temp[:, y, x].reshape(3, 1, 1))
#             # interpolated_img = next_interpolated_img

#     imgs.append(corrected_interpolated_img)
#         # interpolated_img = (1-matrix) * interpolated_img + matrix * (interpolated_img[:, i, j].reshape(3, 1, 1))

# TF.to_pil_image(corrected_interpolated_img).save('./corrected_interpolated_img.jpg')
# make_gif(imgs, save_path="./corrected_interpolated_img.gif")

# label = make_correlation_matrix(interpolated_img.squeeze(0)).unsqueeze(0)

for k in range(100):

    with torch.no_grad():
        imgs.append(transformed_img.clamp_(0, 1).detach().cpu().squeeze(0).clamp_(0, 1))
    
    # pred = feature_extractor(transformed_img)
    # loss = criterion(pred, label_latent_vector)
    # transformed_correlation_matrix = make_correlation_matrix(transformed_img.squeeze(0)).unsqueeze(0)
    
    # loss = criterion(transformed_img, interpolated_img)
    loss = criterion(transformed_img, interpolated_img.unsqueeze(0))

    optim.zero_grad()
    loss.backward(retain_graph=True)
    with torch.no_grad():
        mean_grad = torch.mean(transformed_img.grad)
        expectation_grad = torch.sum(transformed_img.grad * gaussian_matrix.reshape(1, 1, 256, 256).repeat(1, 3, 1, 1))
        for i in range(256):
            for j in range(256):
                if interpolated_mask[:, i, j] >= 0.7:
                    transformed_img.grad[:, :, i, j] = expectation_grad * 0.7
                
                else:
                    transformed_img.grad[:, :, i, j] = (transformed_img.grad[:, :, i, j] + mean_grad) / 2
        # transformed_img.grad[:, :, interpolated_mask]
        # transformed_img.grad = transformed_img.grad * (1 - transformed_mask)
        # print(transformed_img.grad.shape)
        # break
    optim.step()

    print(k, round(loss.item(), 4))

make_gif(imgs, save_path="./corrected_face(v).gif")