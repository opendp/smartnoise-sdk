import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from torchdp import PrivacyEngine, utils, autograd_grad_sample
except ImportError:
    import logging
    test_logger = logging.getLogger(__name__)
    test_logger.warning("requires torch and torchdp")

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, binary=True):
        super(Generator, self).__init__()
        def block(in_, out, Activation):
            return nn.Sequential(
                nn.Linear(in_, out, bias=False),
                nn.LayerNorm(out),
                Activation(),
            )
        
        self.layer_0 = block(latent_dim, latent_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2))
        self.layer_1 = block(latent_dim, latent_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2))
        self.layer_2 = block(latent_dim, output_dim, nn.Tanh if binary else lambda: nn.LeakyReLU(0.2))
        
    def forward(self, noise):
        noise = self.layer_0(noise) + noise
        noise = self.layer_1(noise) + noise
        noise = self.layer_2(noise)
        return noise


class Discriminator(nn.Module):
    def __init__(self, input_dim, wasserstein=False):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2*input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(2*input_dim // 3, input_dim // 3),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 3, 1)
        )

        if not wasserstein:
            self.model.add_module("activation", nn.Sigmoid())

    def forward(self, x):
        return self.model(x)

class DPGAN:
    def __init__(self, 
                 binary=False,
                 latent_dim=64, 
                 batch_size=64,
                 epochs=1000,
                 delta=1e-5,
                 budget=1.0):
        self.binary = binary
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.delta = delta
        self.budget = budget
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, data):
        dataset = TensorDataset(torch.from_numpy(data.astype('float32')).to(self.device))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        self.generator = Generator(self.latent_dim, data.shape[1], binary=self.binary).to(self.device)
        discriminator = Discriminator(data.shape[1]).to(self.device)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=4e-4)
        
        privacy_engine = PrivacyEngine(
            discriminator,
            batch_size=self.batch_size,
            sample_size=len(data),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=3.5,
            max_grad_norm=1.0,
            clip_per_layer=True
        )
        
        privacy_engine.attach(optimizer_d)
        optimizer_g = optim.Adam(self.generator.parameters(), lr=1e-4)

        criterion = nn.BCELoss()
        
        for epoch in range(self.epochs):
            for i, data in enumerate(dataloader):
                discriminator.zero_grad()
                
                real_data = data[0].to(self.device)

                # train with fake data
                noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
                noise = noise.view(-1, self.latent_dim)
                fake_data = self.generator(noise)
                label_fake = torch.full((self.batch_size,), 0, device=self.device)
                output = discriminator(fake_data.detach())
                loss_d_fake = criterion(output, label_fake)
                loss_d_fake.backward()
                optimizer_d.step()
                
                # train with real data
                label_true = torch.full((self.batch_size,), 1, device=self.device)
                output = discriminator(real_data.float())
                loss_d_real = criterion(output, label_true)
                loss_d_real.backward()
                optimizer_d.step()
                
                loss_d = loss_d_real + loss_d_fake

                max_grad_norm = []
                for p in discriminator.parameters():
                    param_norm = p.grad.data.norm(2).item()
                    max_grad_norm.append(param_norm)
                
                privacy_engine.max_grad_norm = max_grad_norm
            
                # train generator
                self.generator.zero_grad()
                label_g = torch.full((self.batch_size,), 1, device=self.device)
                output_g = discriminator(fake_data)
                loss_g = criterion(output_g, label_g)
                loss_g.backward()
                optimizer_g.step()
            
                # manually clear gradients
                for p in discriminator.parameters():
                    if hasattr(p, "grad_sample"):
                        del p.grad_sample

                if self.delta is None:
                    self.delta = 1 / data.shape[0]
                
                epsilon, best_alpha = optimizer_d.privacy_engine.get_privacy_spent(self.delta)
                
            if self.budget < epsilon:
                break

    def generate(self, n):
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
            noise = noise.view(-1, self.latent_dim)
            
            fake_data = self.generator(noise)
            data.append(fake_data.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return data