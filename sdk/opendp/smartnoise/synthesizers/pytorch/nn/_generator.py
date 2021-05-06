import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, binary=True):
        super(Generator, self).__init__()

        def block(in_, out, activation):
            return nn.Sequential(nn.Linear(in_, out, bias=False), nn.LayerNorm(out), activation(),)

        self.layer_0 = block(
            latent_dim, latent_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2)
        )
        self.layer_1 = block(
            latent_dim, latent_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2)
        )
        self.layer_2 = block(
            latent_dim, output_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2)
        )

    def forward(self, noise):
        noise = self.layer_0(noise) + noise
        noise = self.layer_1(noise) + noise
        noise = self.layer_2(noise)
        return noise
