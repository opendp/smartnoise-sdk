import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from opacus import PrivacyEngine

from snsynth.base import Synthesizer

from ._generator import Generator
from ._discriminator import Discriminator


class DPGAN(Synthesizer):
    def __init__(
        self,
        binary=False,
        latent_dim=64,
        batch_size=64,
        epochs=1000,
        delta=None,
        epsilon=1.0,
    ):
        self.binary = binary
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.delta = delta
        self.epsilon = epsilon

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
        elif isinstance(data, list):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array or pandas dataframe")

        dataset = TensorDataset(
            torch.from_numpy(data.astype("float32")).to(self.device)
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        self.generator = Generator(
            self.latent_dim, data.shape[1], binary=self.binary
        ).to(self.device)
        discriminator = Discriminator(data.shape[1]).to(self.device)
        optimizer_d = optim.Adam(discriminator.parameters(), lr=4e-4)

        privacy_engine = PrivacyEngine(
            discriminator,
            batch_size=self.batch_size,
            sample_size=len(data),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=3.5,
            max_grad_norm=1.0,
            clip_per_layer=True,
        )

        privacy_engine.attach(optimizer_d)
        optimizer_g = optim.Adam(self.generator.parameters(), lr=1e-4)

        criterion = nn.BCELoss()

        if self.delta is None:
            self.delta = 1 / (data.shape[0] * np.sqrt(data.shape[0]))

        for epoch in range(self.epochs):
            eps, best_alpha = optimizer_d.privacy_engine.get_privacy_spent(self.delta)

            if self.epsilon < eps:
                if epoch == 0:
                    raise ValueError(
                        "Inputted epsilon and sigma parameters are too small to"
                        + " create a private dataset. Try increasing either parameter and rerunning."
                    )
                break

            for i, data in enumerate(dataloader):
                discriminator.zero_grad()

                real_data = data[0].to(self.device)

                # train with fake data
                noise = torch.randn(
                    self.batch_size, self.latent_dim, 1, 1, device=self.device
                )
                noise = noise.view(-1, self.latent_dim)
                fake_data = self.generator(noise)
                label_fake = torch.full(
                    (self.batch_size,), 0, dtype=torch.float, device=self.device
                )
                output = discriminator(fake_data.detach())
                loss_d_fake = criterion(output.squeeze(), label_fake)
                loss_d_fake.backward()
                optimizer_d.step()

                # train with real data
                label_true = torch.full(
                    (self.batch_size,), 1, dtype=torch.float, device=self.device
                )
                output = discriminator(real_data.float())
                loss_d_real = criterion(output.squeeze(), label_true)
                loss_d_real.backward()
                optimizer_d.step()

                max_grad_norm = []
                for p in discriminator.parameters():
                    param_norm = p.grad.data.norm(2).item()
                    max_grad_norm.append(param_norm)

                privacy_engine.max_grad_norm = max_grad_norm

                # train generator
                self.generator.zero_grad()
                label_g = torch.full(
                    (self.batch_size,), 1, dtype=torch.float, device=self.device
                )
                output_g = discriminator(fake_data)
                loss_g = criterion(output_g.squeeze(), label_g)
                loss_g.backward()
                optimizer_g.step()

                # manually clear gradients
                for p in discriminator.parameters():
                    if hasattr(p, "grad_sample"):
                        del p.grad_sample
                # autograd_grad_sample.clear_backprops(discriminator)

                if self.delta is None:
                    self.delta = 1 / data.shape[0]

    def generate(self, n):
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(
                self.batch_size, self.latent_dim, 1, 1, device=self.device
            )
            noise = noise.view(-1, self.latent_dim)

            fake_data = self.generator(noise)
            data.append(fake_data.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def fit(self, data, *ignore, transformer=None, categorical_columns=[], ordinal_columns=[], continuous_columns=[], preprocessor_eps=0.0, nullable=False):
        self.train(data, transformer=transformer, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns, preprocessor_eps=preprocessor_eps, nullable=nullable)

    def sample(self, n_samples):
        return self.generate(n_samples)
