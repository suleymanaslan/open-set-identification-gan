import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_size=100, input_channels=3, feature_map_size=64):
        super(Generator, self).__init__()
        net_channels = [latent_size,
                        feature_map_size*8,
                        feature_map_size*4,
                        feature_map_size*2,
                        feature_map_size*1,
                        input_channels]
        self.main = nn.Sequential(
            nn.ConvTranspose2d(net_channels[0], net_channels[1], 4, 1, 0), nn.BatchNorm2d(net_channels[1]), nn.ReLU(True),
            nn.ConvTranspose2d(net_channels[1], net_channels[2], 4, 2, 1), nn.BatchNorm2d(net_channels[2]), nn.ReLU(True),
            nn.ConvTranspose2d(net_channels[2], net_channels[3], 4, 2, 1), nn.BatchNorm2d(net_channels[3]), nn.ReLU(True),
            nn.ConvTranspose2d(net_channels[3], net_channels[4], 4, 2, 1), nn.BatchNorm2d(net_channels[4]), nn.ReLU(True),
            nn.ConvTranspose2d(net_channels[4], net_channels[5], 4, 2, 1), nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, feature_map_size=64, groups=1):
        super(Discriminator, self).__init__()
        net_channels = [input_channels,
                        feature_map_size*1,
                        feature_map_size*2,
                        feature_map_size*4,
                        feature_map_size*8,
                        output_channels]
        self.main = nn.Sequential(
            nn.Conv2d(net_channels[0], net_channels[1], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[1], net_channels[2], 4, 2, 1), nn.BatchNorm2d(net_channels[2]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[2], net_channels[3], 4, 2, 1), nn.BatchNorm2d(net_channels[3]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[3], net_channels[4], 4, 2, 1), nn.BatchNorm2d(net_channels[4]), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(net_channels[4], net_channels[5], 4, 1, 0), nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
