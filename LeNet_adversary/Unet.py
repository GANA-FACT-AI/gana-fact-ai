import torch
import torch.nn as nn


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, input_dim, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers = list()
        layers.extend([
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ])
        layers.extend([
                          nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(inplace=True)
                      ] * 4)
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, input_dim):
        super(Up, self).__init__()
        self.transpose_conv_1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            kernel_size=2)

        self.up_conv_1 = Down(in_channels, out_channels, input_dim)

    def forward(self, init_tensor, x):
        """
        Init_tensor is a tensor from earlier layers which is croped and added to the upscaled features
        """
        x1 = self.transpose_conv_1(x)
        x2 = crop_image(init_tensor, x1)
        x3 = self.up_conv_1(torch.cat([x1, x2],1))
        return x3



def crop_image(tensor1, tensor2):
    input_size =  tensor1.shape[2]
    target_size = tensor2.shape[2]
    diff = (input_size - target_size)//2
    return tensor1[:,:, diff: input_size - diff, diff: input_size - diff]


class UNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels):

        super().__init__()
        self.n_channels = n_input_channels
        self.n_classes = n_output_channels

        self.donw_conv_1 = Down(n_input_channels, 64, 32)
        self.down_conv_2 = Down(64, 128, 16)
        self.down_conv_3 = Down(128, 256, 8)
        self.down_conv_4 = Down(256, 512, 4)
        self.down_conv_5 = Down(512, 1024, 2)

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.transpose_conv_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            stride=2,
            kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2.3, mode='bilinear')

        self.up_1 = Up(1024, 512, 2)
        self.up_2 = Up(512, 256, 4)
        self.up_3 = Up(256, 128, 8)
        self.up_4 = Up(128, 64, 16)

        self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.upsample(x)
        x1 = self.donw_conv_1(x0)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        x = self.up_1(x7, x9)
        x = self.up_2(x5, x)
        x = self.up_3(x3, x)
        x = self.up_4(x1, x)
        x = self.out(x)

        return x