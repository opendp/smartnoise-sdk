import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

from opendp.whitenoise.synthesizers.utils.pate import weights_init, pate, moments_acc

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Generator, self).__init__()
        previous_layer_dim = latent_dim
        
        model = []
        for i, layer_dim in enumerate(list(hidden_dims)):
            model.append(nn.Linear(previous_layer_dim, layer_dim))
            model.append(nn.BatchNorm1d(layer_dim))
            model.append(nn.ReLU())
            previous_layer_dim = layer_dim
        model.append(nn.Linear(previous_layer_dim, output_dim))
        
        self.model = nn.Sequential(*model)
    
    def forward(self, noise):
        data = self.model(noise)
        return data

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Discriminator, self).__init__()
        previous_layer_dim = input_dim
        
        model = []
        if isinstance(hidden_dims, tuple):
            for i, layer_dim in enumerate(list(hidden_dims)):
                model.append(nn.Linear(previous_layer_dim, layer_dim))
                model.append(nn.LeakyReLU(0.2))
                model.append(nn.Dropout(0.5))
                previous_layer_dim = layer_dim
        else:
            model.append(nn.Linear(previous_layer_dim, hidden_dims))
            previous_layer_dim = hidden_dims
        model.append(nn.Linear(previous_layer_dim, 1))
        model.append(nn.Sigmoid())

        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        output = self.model(input)
        return output.view(-1)

class PATEGANSynthesizer:
    def __init__(self,
                 latent_dim=32,
                 gen_dim=(128, 64, 128),
                 batch_size=64,
                 teacher_iters=5,
                 student_iters=5,
                 budget=3.0,
                 delta=1e-5):
        self.latent_dim = latent_dim
        self.gen_dim = gen_dim
        self.batch_size = batch_size
        self.teacher_iters = teacher_iters
        self.student_iters = student_iters
        self.budget = budget
        self.delta = delta

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        data_dim = data.shape[1]

        self.num_teachers = int(len(data) / 1000)
        
        data_partitions = np.array_split(data, self.num_teachers)
        tensor_partitions = [TensorDataset(torch.from_numpy(data.astype('double')).to(self.device)) for data in data_partitions]
        
        loader = []
        for teacher_id in range(self.num_teachers):
            loader.append(DataLoader(tensor_partitions[teacher_id], batch_size=self.batch_size, shuffle=True))

        self.generator = Generator(self.latent_dim, self.gen_dim, data_dim).double().to(self.device)
        self.generator.apply(weights_init)
        
        student_dim = (data_dim, int(data_dim/2), data_dim)
        student_disc = Discriminator(data_dim, student_dim).double().to(self.device)
        student_disc.apply(weights_init)

        teacher_disc = [Discriminator(data_dim, data_dim).double().to(self.device) for i in range(self.num_teachers)]
        for i in range(self.num_teachers):
            teacher_disc[i].apply(weights_init)
        
        optimizer_g = optim.Adam(self.generator.parameters(), lr=1e-4)
        optimizer_s = optim.Adam(student_disc.parameters(), lr=1e-4)
        optimizer_t = [optim.Adam(teacher_disc[i].parameters(), lr=1e-4) for i in range(self.num_teachers)]

        criterion = nn.BCELoss()

        noise_multiplier = 1e-3
        alphas = torch.tensor([0.0 for i in range(100)])
        l_list = 1 + torch.tensor(range(100))
        epsilon = 0

        while epsilon < self.budget:
            
            # train teacher discriminators
            for t_2 in range(self.teacher_iters):
                for i in range(self.num_teachers):
                    real_data, category = None, None
                    for j, data in enumerate(loader[i], 0):
                        real_data = data[0].to(self.device)
                        break
                    
                    optimizer_t[i].zero_grad()

                    # train with real data
                    label_real = torch.full((real_data.shape[0],), 1, device=self.device)
                    output = teacher_disc[i](real_data)
                    loss_t_real = criterion(output, label_real.double())
                    loss_t_real.backward()

                    # train with fake data
                    noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
                    label_fake = torch.full((self.batch_size,), 0, device=self.device)
                    fake_data = self.generator(noise.double())
                    output = teacher_disc[i](fake_data)
                    loss_t_fake = criterion(output, label_fake.double())
                    loss_t_fake.backward()
                    optimizer_t[i].step()

            # train student discriminator
            for t_3 in range(self.student_iters):
                noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(noise.double())
                predictions, votes = pate(fake_data, teacher_disc, noise_multiplier)
                output = student_disc(fake_data.detach())

                # update moments accountant
                alphas = alphas + moments_acc(self.num_teachers, votes, noise_multiplier, l_list)

                loss_s = criterion(output, predictions.to(self.device))
                loss_s.backward()
                optimizer_s.step()

            # train generator
            label_g = torch.full((self.batch_size,), 1, device=self.device)
            noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
            gen_data = self.generator(noise.double())
            output_g = student_disc(gen_data)
            loss_g = criterion(output_g, label_g.double())
            loss_g.backward()
            optimizer_g.step()

            epsilon = min((alphas - math.log(self.delta)) / l_list)

    def sample(self, n):
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            noise = noise.view(-1, self.latent_dim)
            
            fake_data = self.generator(noise.double())
            data.append(fake_data.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return data
