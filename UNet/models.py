import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class UpSampling(nn.Module):

    def __init__(self, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor

    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)


def encoder_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),

        nn.MaxPool2d(2, 2)
    )


def decoder_block(in_channels, out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        UpSampling(scale_factor=2.0),
        nn.BatchNorm2d(out_channels),
        nn.GELU()
        # nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    )



class UNet(nn.Module):
    
    def __init__(self):
        super().__init__()

        # input shape: (3, 256, 256)
        self.encoder = nn.ModuleList([
            encoder_block(in_channels=3, out_channels=16), # (16, 128, 128)
            encoder_block(in_channels=16, out_channels=32), # (32, 64, 64)
            encoder_block(in_channels=32, out_channels=64), # (64, 32, 32)
            encoder_block(in_channels=64, out_channels=128), # (128, 16, 16)
            encoder_block(in_channels=128, out_channels=256), # (256, 8, 8)
        ])

        self.center = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1), # (512, 4, 4)
                nn.BatchNorm2d(512),
                nn.GELU()
            ),

            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (512, 4, 4)
                nn.BatchNorm2d(512),
                nn.GELU()
            ),

            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1), # (256, 4, 4)
                UpSampling(scale_factor=2.0), # (256, 8, 8)
                nn.BatchNorm2d(256),
                nn.GELU()
            )
        ])

        self.decoder = nn.ModuleList([
            decoder_block(in_channels=256+256, out_channels=256), # (256, 16, 16)
            decoder_block(in_channels=256+128, out_channels=128), # (128, 32, 32)
            decoder_block(in_channels=128+64, out_channels=64), # (64, 64, 64)
            decoder_block(in_channels=64+32, out_channels=32), # (32, 128, 128)
        ])

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=32+16, out_channels=16, kernel_size=3, stride=1, padding=1), # (16, 128, 128)
            UpSampling(scale_factor=2.0), # (16, 256, 256)
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1), # (16, 256, 256)
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0), # (1, 256, 256)
            nn.Sigmoid()
        )

    
    def forward(self, x):
        encodings = {
            -1: x
        }

        for i in range(len(self.encoder)):
            encodings[i] = self.encoder[i](encodings[i-1])

        out = encodings[i]
        for i in range(len(self.center)):
            out = self.center[i](out)

        # decoder_pair[decoder_index] = encoder_index
        decoder_pair = [4, 3, 2, 1]

        for i in range(len(self.decoder)):
            encoder_idx = decoder_pair[i]

            out = torch.concat((out, encodings[encoder_idx]), dim=1) # (batch_size, C, H, W)
            out = self.decoder[i](out)

        out = torch.concat((out, encodings[0]), dim=1)
        out = self.final(out)

        return out


if __name__ == '__main__':
    summary(UNet().cuda(), (3, 256, 256))