import torch
from torch.nn import MSELoss
from torch import optim

from PIL import Image

import torchvision.transforms.functional as TF

import sys
sys.path.append("/home/kdhsimplepro/kdhsimplepro/AI/Stylers/UNet")

from models import UNet


unet = UNet().to("cuda")

unet.load_state_dict(torch.load("/home/kdhsimplepro/kdhsimplepro/AI/Stylers/UNet/male_female_best_params.pth"))
unet.eval()

@torch.no_grad()
def pred_unet(x):

    mask = unet(x)
    return mask

expr = 5
expr_dir = f"./test_expr{expr}"
original_paths = ["./v_0.png", "./v_1.png", "./jungwoon/ai_profile.png", "./jungwoon/in_the_car.png", "./jungwoon/mirror.png"]
transformed_paths = [f"{expr_dir}/v_0_test.png", f"{expr_dir}/v_1_test.png", f"{expr_dir}/jungwoon/ai_profile_test.png", f"{expr_dir}/jungwoon/in_the_car_test.png", f"{expr_dir}/jungwoon/mirror_test.png"]
save_paths = [f"{expr_dir}/corrected_face_v_0_test.gif", f"{expr_dir}/corrected_face_v_1_test.gif", f"{expr_dir}/jungwoon/corrected_face_ai_profile.gif", f"{expr_dir}/jungwoon/corrected_face_in_the_car.gif", f"{expr_dir}/jungwoon/corrected_face_mirror.gif"]

idx = 4
original_path = original_paths[idx]
transformed_path = transformed_paths[idx]
save_path = save_paths[idx]

original_img = TF.to_tensor(Image.open(original_path).resize((256, 256)).convert("RGB")).to("cuda").unsqueeze(0)
transformed_img = TF.to_tensor(Image.open(transformed_path).resize((256, 256)).convert("RGB")).to("cuda").unsqueeze(0)
transformed_img.requires_grad = True

original_mask = pred_unet(original_img)
transformed_mask = pred_unet(transformed_img).detach()

optimizer = optim.Adam([transformed_img], lr=0.1)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1)

label = original_img * (1 - original_mask)

criterion = MSELoss()

imgs = []


interpolated_mask = (original_mask + transformed_mask).clamp_(0, 1).cuda().squeeze(0)

def interpolate_img(original_img, transformed_img, interpolated_mask):
    # interpolated_img = original_img * (1-interpolated_mask) + interpolated_mask * transformed_img
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

    dist = torch.distributions.normal.Normal(torch.tensor([0]), torch.tensor([size/2]))
    distance = torch.sum(torch.abs(coordinates - torch.Tensor([y, x])), dim=2)
    matrix = torch.exp(dist.log_prob(distance))
    matrix /= torch.sum(matrix)
    
    return matrix

gaussian_matrix_dict = {}
weight_dict = {}

for i in range(256):
    for j in range(256):
        if interpolated_mask[0, i, j] >= 0.7:
            gaussian_matrix_dict[(i, j)] = get_gaussian_dist(i, j, 256).to("cuda")
            weight_dict[(i, j)] = transformed_img[:, :, i, j] / (torch.sum(transformed_img * gaussian_matrix_dict[(i, j)].reshape(1, 1, 256, 256).repeat(1, 3, 1, 1)))

# skin_color = torch.sum(original_img * get_gaussian_dist(128, 128, 256).unsqueeze(0).to("cuda"), dim=(2, 3)).reshape(-1)
# skin_gaussian_matrix = get_gaussian_dist(128, 128, 256).unsqueeze(0).to("cuda")

interpolated_img = interpolated_img.squeeze(0)

alpha = 0.01
for k in range(50):

    with torch.no_grad():
        imgs.append(transformed_img.clamp_(0, 1).detach().cpu().squeeze(0).clamp_(0, 1))

    loss = criterion(transformed_img, interpolated_img.unsqueeze(0))

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    with torch.no_grad():
        mean_grad = torch.mean(transformed_img.grad)
        
        for i in range(256):
            for j in range(256):
                if interpolated_mask[:, i, j] >= 0.7:
                    transformed_img.grad[:, :, i, j] = mean_grad * (1-alpha)

                else:
                    transformed_img.grad[:, :, i, j] = (alpha * transformed_img.grad[:, :, i, j] + (1-alpha)*mean_grad)

    
    optimizer.step()
    scheduler.step()

    # with torch.no_grad():
    #     for i in range(256):
    #         for j in range(256):
    #             if interpolated_mask[:, i, j] >= 0.7:
    #                 # if transformed_mask[:, :, i, j] < 0.7:
    #                 transformed_img[:, :, i, j] = weight_dict[(i, j)] * (torch.sum(transformed_img * gaussian_matrix_dict[(i, j)].reshape(1, 1, 256, 256).repeat(1, 3, 1, 1)))

    print(k, round(loss.item(), 4))

# deleted_mask = (original_mask - transformed_mask).clamp_(0, 1)
# mean_ = torch.mean(transformed_img * deleted_mask, dim=(2, 3)).reshape(-1) - skin_color
# # mean_ = torch.mean(transformed_img[0][interpolated_mask >= 0.7]) - skin_color
# with torch.no_grad():
#     for i in range(256):
#         for j in range(256):
#             if deleted_mask[:, :, i, j] >= 0.7:
#                 transformed_img[:, :, i, j] -= mean_
#                 transformed_img[:, :, i, j] = transformed_img[:, :, i, j].clamp_(0, 1)
        
#         imgs.append(transformed_img.clamp_(0, 1).detach().cpu().squeeze(0).clamp_(0, 1))

# make_gif(imgs, save_path="./test_expr7/jungwoon/corrected_face_ai_profile.gif")
make_gif(imgs, save_path=save_path)
print(save_path)