import math
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch import nn
import torch.utils.data
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.autograd import Variable
import warnings

from snsynth.base import Synthesizer

from .ctgan.data_sampler import DataSampler
from .ctgan.ctgan import CTGANSynthesizer
from snsynth.transform.table import TableTransformer

from .privacy_utils import weights_init, pate, moments_acc


class Discriminator(Module):
    def __init__(self, input_dim, discriminator_dim, loss, pac=10):
        super(Discriminator, self).__init__()
        torch.cuda.manual_seed(0)
        torch.manual_seed(0)

        dim = input_dim * pac
        #  print ('now dim is {}'.format(dim))
        self.pac = pac
        self.pacdim = dim

        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        if loss == "cross_entropy":
            seq += [Sigmoid()]
        self.seq = Sequential(*seq)

    def dragan_penalty(self, real_data, device="cpu", pac=10, lambda_=10):
        # real_data = torch.from_numpy(real_data).to(device)
        alpha = (
            torch.rand(real_data.shape[0], 1, device=device)
            .squeeze()
            .expand(real_data.shape[0])
        )
        delta = torch.normal(
            mean=0.0, std=float(pac), size=real_data.shape, device=device
        )  # 0.5 * real_data.std() * torch.rand(real_data.shape)
        x_hat = Variable(
            (alpha * real_data.T + (1 - alpha) * (real_data + delta).T).T,
            requires_grad=True,
        )

        pred_hat = self(x_hat.float())

        gradients = torch.autograd.grad(
            outputs=pred_hat,
            inputs=x_hat,
            grad_outputs=torch.ones(pred_hat.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        dragan_penalty = lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return dragan_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))


class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):
    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


class PATECTGAN(CTGANSynthesizer, Synthesizer):
    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        verbose=True,
        epochs=300,
        pac=1,
        cuda=True,
        epsilon=1,
        binary=False,
        regularization=None,
        loss="cross_entropy",
        teacher_iters=5,
        student_iters=5,
        sample_per_teacher=1000,
        delta=None,
        noise_multiplier=1e-3,
        moments_order=100,
    ):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self.epsilon = epsilon
        self.verbose = verbose
        self.loss = loss

        # PATE params
        self.regularization = regularization if self.loss != "wasserstein" else "dragan"
        self.teacher_iters = teacher_iters
        self.student_iters = student_iters
        self.pd_cols = None
        self.pd_index = None
        self.binary = binary
        self.sample_per_teacher = sample_per_teacher
        self.noise_multiplier = noise_multiplier
        self.moments_order = moments_order
        self.delta = delta

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

    def train(
        self,
        data,
        categorical_columns=None,
        ordinal_columns=None,
        update_epsilon=None,
        transformer=None,
        continuous_columns=None, 
        preprocessor_eps=0.0,
        nullable=False
    ):
        if update_epsilon:
            self.epsilon = update_epsilon

        sample_per_teacher = (
            self.sample_per_teacher if self.sample_per_teacher < len(data) else 1000
        )
        self.num_teachers = int(len(data) / sample_per_teacher) + 1

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

        train_data = np.array([
            [float(x) if x is not None else 0.0 for x in row] for row in train_data
        ])

        data_partitions = np.array_split(train_data, self.num_teachers)

        data_dim = self._transformer.output_width


        self.cond_generator = DataSampler(
            train_data,
            self._transformer.transformers
        )

        cached_probs = self.cond_generator.discrete_column_category_prob

        cond_generator = [
            DataSampler(
                d,
                self._transformer.transformers,
                discrete_column_category_prob=cached_probs,
            )
            for d in data_partitions
        ]

        self._generator = Generator(
            self._embedding_dim + self.cond_generator.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self.cond_generator.dim_cond_vec(),
            self._discriminator_dim,
            self.loss,
            self.pac,
        ).to(self._device)

        student_disc = discriminator
        student_disc.apply(weights_init)

        teacher_disc = [discriminator for i in range(self.num_teachers)]
        for i in range(self.num_teachers):
            teacher_disc[i].apply(weights_init)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizer_s = optim.Adam(student_disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
        optimizer_t = [
            optim.Adam(
                teacher_disc[i].parameters(),
                lr=self._discriminator_lr,
                betas=(0.5, 0.9),
                weight_decay=self._discriminator_decay,
            )
            for i in range(self.num_teachers)
        ]

        noise_multiplier = self.noise_multiplier
        alphas = torch.tensor(
            [0.0 for i in range(self.moments_order)], device=self._device
        )
        l_list = 1 + torch.tensor(range(self.moments_order), device=self._device)
        eps = torch.zeros(1)

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        real_label = 1
        fake_label = 0

        criterion = nn.BCELoss() if (self.loss == "cross_entropy") else self.w_loss

        if self.verbose:
            print(
                "using loss {} and regularization {}".format(
                    self.loss, self.regularization
                )
            )

        iteration = 0

        if self.delta is None:
            self.delta = 1 / (train_data.shape[0] * np.sqrt(train_data.shape[0]))

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
                    partition_data = data_partitions[i]
                    data_sampler = DataSampler(
                        partition_data,
                        self._transformer.transformers,
                        discrete_column_category_prob=cached_probs,
                    )
                    fakez = torch.normal(mean, std=std).to(self._device)

                    condvec = cond_generator[i].sample_condvec(self._batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)
                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype("float32")).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fake

                    optimizer_t[i].zero_grad()

                    y_all = torch.cat(
                        [teacher_disc[i](fake_cat), teacher_disc[i](real_cat)]
                    )
                    label_fake = torch.full(
                        (int(self._batch_size / self.pac), 1),
                        fake_label,
                        dtype=torch.float,
                        device=self._device,
                    )
                    label_true = torch.full(
                        (int(self._batch_size / self.pac), 1),
                        real_label,
                        dtype=torch.float,
                        device=self._device,
                    )
                    labels = torch.cat([label_fake, label_true])

                    error_d = criterion(y_all.squeeze(), labels.squeeze())
                    error_d.backward()

                    if self.regularization == "dragan":
                        pen = teacher_disc[i].dragan_penalty(
                            real_cat, device=self._device
                        )
                        pen.backward(retain_graph=True)

                    optimizer_t[i].step()
            ###
            # train student discriminator
            for t_3 in range(self.student_iters):
                data_sampler = DataSampler(
                    train_data,
                    self._transformer.transformers,
                    discrete_column_category_prob=cached_probs,
                )
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample_data(self._batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample_data(
                        self._batch_size, col[perm], opt[perm]
                    )
                    c2 = c1[perm]

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                else:
                    fake_cat = fakeact

                fake_data = fake_cat

                ###
                predictions, votes = pate(
                    fake_data, teacher_disc, noise_multiplier, device=self._device
                )

                output = student_disc(fake_data.detach())

                # update moments accountant
                alphas = alphas + moments_acc(
                    self.num_teachers,
                    votes,
                    noise_multiplier,
                    l_list,
                    device=self._device,
                )

                loss_s = criterion(
                    output.squeeze(), predictions.float().to(self._device).squeeze()
                )

                optimizer_s.zero_grad()
                loss_s.backward()

                if self.regularization == "dragan":
                    vals = torch.cat([predictions, fake_data], axis=1)
                    ordered = vals[vals[:, 0].sort()[1]]
                    data_list = torch.split(
                        ordered, predictions.shape[0] - int(predictions.sum().item())
                    )
                    synth_cat = torch.cat(data_list[1:], axis=0)[:, 1:]
                    pen = student_disc.dragan_penalty(synth_cat, device=self._device)
                    pen.backward(retain_graph=True)

                optimizer_s.step()

                # print ('iterator {i}, student discriminator loss is {j}'.format(i=t_3, j=loss_s))

            # train generator
            fakez = torch.normal(mean=mean, std=std)
            condvec = self.cond_generator.sample_condvec(self._batch_size)

            if condvec is None:
                c1, m1, col, opt = None, None, None, None
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                m1 = torch.from_numpy(m1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)

            if c1 is not None:
                y_fake = student_disc(torch.cat([fakeact, c1], dim=1))
            else:
                y_fake = student_disc(fakeact)

            if condvec is None:
                cross_entropy = 0
            else:
                cross_entropy = self._cond_loss(fake, c1, m1)

            if self.loss == "cross_entropy":
                label_g = torch.full(
                    (int(self._batch_size / self.pac), 1),
                    real_label,
                    dtype=torch.float,
                    device=self._device,
                )
                loss_g = criterion(y_fake.squeeze(), label_g.float().squeeze())
                loss_g = loss_g + cross_entropy
            else:
                loss_g = -torch.mean(y_fake) + cross_entropy

            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()

            if self.verbose:
                print(
                    "eps: {:f} \t G: {:f} \t D: {:f}".format(
                        eps, loss_g.detach().cpu(), loss_s.detach().cpu()
                    )
                )

    def w_loss(self, output, labels):
        vals = torch.cat([labels[None, :], output[None, :]], axis=1)
        ordered = vals[vals[:, 0].sort()[1]]
        data_list = torch.split(ordered, labels.shape[0] - int(labels.sum().item()), dim=1)
        fake_score = data_list[0][:, 1]
        true_score = torch.cat(data_list[1:], axis=0)[:, 1]
        w_loss = -(torch.mean(true_score) - torch.mean(fake_score))
        return w_loss

    def generate(self, n, condition_column=None, condition_value=None):
        """
        TODO: Add condition_column support 
        """
        self._generator.eval()

        # output_info = self._transformer.output_info
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            condvec = self.cond_generator.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def fit(self, data, *ignore, transformer=None, categorical_columns=[], ordinal_columns=[], continuous_columns=[], preprocessor_eps=0.0, nullable=False):
        self.train(data, transformer=transformer, categorical_columns=categorical_columns, ordinal_columns=ordinal_columns, continuous_columns=continuous_columns, preprocessor_eps=preprocessor_eps, nullable=nullable)

    def sample(self, n_samples):
        return self.generate(n_samples)
