import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from snsynth.base import Synthesizer

from ._generator import Generator
from ._discriminator import Discriminator

from .privacy_utils import weights_init, pate, moments_acc


class PATEGAN(Synthesizer):
    def __init__(
        self,
        epsilon,
        delta=None,
        binary=False,
        latent_dim=64,
        batch_size=64,
        teacher_iters=5,
        student_iters=5,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.binary = binary
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.teacher_iters = teacher_iters
        self.student_iters = student_iters

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pd_cols = None
        self.pd_index = None

    def train(
        self,
        data,
        categorical_columns=None,
        ordinal_columns=None,
        update_epsilon=None,
        transformer=None,
        continuous_columns=None,
        preprocessor_eps=0.0,
        nullable=False,
    ):
        if update_epsilon:
            self.epsilon = update_epsilon

        train_data = self._get_train_data(
            data,
            style='gan',
            transformer=transformer,
            categorical_columns=categorical_columns, 
            ordinal_columns=ordinal_columns, 
            continuous_columns=continuous_columns, 
            nullable=nullable,
            preprocessor_eps=preprocessor_eps
        )

        data = np.array(train_data)

        if isinstance(data, pd.DataFrame):
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="ignore")
            self.pd_cols = data.columns
            self.pd_index = data.index
            data = data.to_numpy()
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array or pandas dataframe")

        data_dim = data.shape[1]

        self.num_teachers = int(len(data) / 1000)

        data_partitions = np.array_split(data, self.num_teachers)
        tensor_partitions = [
            TensorDataset(torch.from_numpy(data.astype("double")).to(self.device))
            for data in data_partitions
        ]

        loader = []
        for teacher_id in range(self.num_teachers):
            loader.append(
                DataLoader(
                    tensor_partitions[teacher_id],
                    batch_size=self.batch_size,
                    shuffle=True,
                )
            )

        self.generator = (
            Generator(self.latent_dim, data_dim, binary=self.binary)
            .double()
            .to(self.device)
        )
        self.generator.apply(weights_init)

        student_disc = Discriminator(data_dim).double().to(self.device)
        student_disc.apply(weights_init)

        teacher_disc = [
            Discriminator(data_dim).double().to(self.device)
            for i in range(self.num_teachers)
        ]
        for i in range(self.num_teachers):
            teacher_disc[i].apply(weights_init)

        optimizer_g = optim.Adam(self.generator.parameters(), lr=1e-4)
        optimizer_s = optim.Adam(student_disc.parameters(), lr=1e-4)
        optimizer_t = [
            optim.Adam(teacher_disc[i].parameters(), lr=1e-4)
            for i in range(self.num_teachers)
        ]

        criterion = nn.BCELoss()

        noise_multiplier = 1e-3
        alphas = torch.tensor([0.0 for i in range(100)])
        l_list = 1 + torch.tensor(range(100))
        eps = torch.zeros(1)

        if self.delta is None:
            self.delta = 1 / (data.shape[0] * np.sqrt(data.shape[0]))

        iteration = 0
        while eps.item() < self.epsilon:
            iteration += 1

            eps = min((alphas - math.log(self.delta)) / l_list)

            if eps.item() > self.epsilon:
                if iteration == 1:
                    raise ValueError(
                        "Inputted epsilon parameter is too small to"
                        + " create a private dataset. Try increasing epsilon and rerunning."
                    )
                break

            # train teacher discriminators
            for t_2 in range(self.teacher_iters):
                for i in range(self.num_teachers):
                    real_data = None
                    for j, data in enumerate(loader[i], 0):
                        real_data = data[0].to(self.device)
                        break

                    optimizer_t[i].zero_grad()

                    # train with real data
                    label_real = torch.full(
                        (real_data.shape[0],), 1, dtype=torch.float, device=self.device
                    )
                    output = teacher_disc[i](real_data)
                    loss_t_real = criterion(output.squeeze(), label_real.double())
                    loss_t_real.backward()

                    # train with fake data
                    noise = torch.rand(
                        self.batch_size, self.latent_dim, device=self.device
                    )
                    label_fake = torch.full(
                        (self.batch_size,), 0, dtype=torch.float, device=self.device
                    )
                    fake_data = self.generator(noise.double())
                    output = teacher_disc[i](fake_data)
                    loss_t_fake = criterion(output.squeeze(), label_fake.double())
                    loss_t_fake.backward()
                    optimizer_t[i].step()

            # train student discriminator
            for t_3 in range(self.student_iters):
                noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(noise.double())
                predictions, votes = pate(fake_data, teacher_disc, noise_multiplier)
                output = student_disc(fake_data.detach())

                # update moments accountant
                alphas = alphas + moments_acc(
                    self.num_teachers, votes, noise_multiplier, l_list
                )

                loss_s = criterion(
                    output.squeeze(), predictions.to(self.device).squeeze()
                )
                optimizer_s.zero_grad()
                loss_s.backward()
                optimizer_s.step()

            # train generator
            label_g = torch.full(
                (self.batch_size,), 1, dtype=torch.float, device=self.device
            )
            noise = torch.rand(self.batch_size, self.latent_dim, device=self.device)
            gen_data = self.generator(noise.double())
            output_g = student_disc(gen_data)
            loss_g = criterion(output_g.squeeze(), label_g.double())
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

    def generate(self, n):
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(self.batch_size, self.latent_dim, device=self.device)
            noise = noise.view(-1, self.latent_dim)

            fake_data = self.generator(noise.double())
            data.append(fake_data.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def fit(self, data, *ignore, transformer=None, categorical_columns=[], ordinal_columns=[], continuous_columns=[], preprocessor_eps=0.0, nullable=False):
        self.train(data, transformer=transformer, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns, preprocessor_eps=preprocessor_eps, nullable=nullable)

    def sample(self, n_samples):
        return self.generate(n_samples)
