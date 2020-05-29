import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchdp import PrivacyEngine, utils, autograd_grad_sample
from .utils import GeneralTransformer

from opendp.whitenoise.synthesizers.base import SDGYMBaseSynthesizer

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Generator, self).__init__()
        
        hidden_activation = nn.ReLU()
        previous_layer_dim = latent_dim
        
        model = []
        for i, layer_dim in enumerate(list(hidden_dims)):
            model.append(nn.Linear(previous_layer_dim, layer_dim))
            model.append(nn.BatchNorm1d(layer_dim))
            model.append(hidden_activation)
            previous_layer_dim = layer_dim
        model.append(nn.Linear(previous_layer_dim, output_dim))
        
        self.model = nn.Sequential(*model)
    
    def forward(self, noise):
        data = self.model(noise)
        return data

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims, pac=4):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pac_dim = dim
        
        model = []
        for i, layer_dim in enumerate(list(hidden_dims)):
            model.append(nn.Linear(dim, layer_dim))
            model.append(nn.LeakyReLU(0.2))
            model.append(nn.Dropout(0.5))
            dim = layer_dim
        model.append(nn.Linear(dim, 1))
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input.view(-1, self.pac_dim))

def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
        else:
            continue
    return torch.cat(data_t, dim=1)

class DPGANSynthesizer(SDGYMBaseSynthesizer):
    def __init__(self,
                 latent_dim=128,
                 gen_dim=(256, 256),
                 dis_dim=(256, 256),
                 batch_size=256,
                 epochs=1000,
                 n_critics=5, 
                 delta=1e-5,
                 budget=3.0):
        self.latent_dim = latent_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_critics = n_critics
        self.delta = delta
        self.budget = budget
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.transformer = GeneralTransformer()
        self.transformer.fit(data, categorical_columns, ordinal_columns)
        data = self.transformer.transform(data)
        dataset = TensorDataset(torch.from_numpy(data.astype('float32')).to(self.device))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        data_dim = self.transformer.output_dim
        
        self.generator = Generator(self.latent_dim, self.gen_dim, data_dim)
        self.generator = utils.convert_batchnorm_modules(self.generator).to(self.device)
        
        discriminator = Discriminator(data_dim, self.dis_dim)
        discriminator = utils.convert_batchnorm_modules(discriminator).to(self.device)
        
        optimizer_d = optim.RMSprop(discriminator.parameters(), lr=4e-4)
        
        privacy_engine = PrivacyEngine(
            discriminator,
            batch_size=self.batch_size,
            sample_size=len(data),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=5.0,
            max_grad_norm=1.0,
            clip_per_layer=True
        )
        
        privacy_engine.attach(optimizer_d)
        optimizer_g = optim.RMSprop(self.generator.parameters(), lr=1e-4)
        
        for epoch in range(self.epochs):
            for i, data in enumerate(dataloader):
                discriminator.zero_grad()
                
                real_data = data[0].to(self.device)
                
                # train with fake data
                noise = torch.randn(self.batch_size, self.latent_dim, 1, 1)
                noise = noise.view(-1, self.latent_dim)
                fake_data = self.generator(noise)
                fake_data = apply_activate(fake_data, self.transformer.output_info)
                output = discriminator(fake_data)
                loss_d_fake = torch.mean(output)
                loss_d_fake.backward()
                optimizer_d.step()
                
                # train with real data
                output = discriminator(real_data.float())
                loss_d_real = torch.mean(output)
                loss_d_real.backward()
                optimizer_d.step()
                
                loss_d = -loss_d_real + loss_d_fake

                max_grad_norm = []
                for p in discriminator.parameters():
                    param_norm = p.grad.data.norm(2).item()
                    max_grad_norm.append(param_norm)
                
                privacy_engine.max_grad_norm = max_grad_norm
            
                # train generator
                if i % self.n_critics == 0:
                    self.generator.zero_grad()

                    gen_data = self.generator(noise)
                    loss_g = -torch.mean(discriminator(gen_data))
                    
                    loss_g.backward()
                    optimizer_g.step()
                
                # manually clear gradients
                autograd_grad_sample.clear_backprops(discriminator)
                
                epsilon, best_alpha = optimizer_d.privacy_engine.get_privacy_spent(self.delta)
                
            if self.budget < epsilon:
                break
    
    def sample(self, n):
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(self.batch_size, self.latent_dim, 1, 1)
            noise = noise.view(-1, self.latent_dim)
            
            fake_data = self.generator(noise)
            fake_data = apply_activate(fake_data, self.transformer.output_info)
            data.append(fake_data.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self.transformer.inverse_transform(data, None)