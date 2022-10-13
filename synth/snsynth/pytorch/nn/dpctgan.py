from collections import namedtuple
import numpy as np
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
import warnings

import opacus
from snsynth.base import Synthesizer

from snsynth.transform.table import TableTransformer
from .ctgan.data_sampler import DataSampler
from .ctgan.ctgan import CTGANSynthesizer


class Discriminator(Module):
    def __init__(self, input_dim, discriminator_dim, loss, pac=1):
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

    def calc_gradient_penalty(
        self, real_data, fake_data, device="cpu", pac=1, lambda_=10
    ):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (
            (gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2
        ).mean() * lambda_

        return gradient_penalty

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


# custom for calcuate grad_sample for multiple loss.backward()
def _custom_create_or_extend_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int
) -> None:
    """
    Create a 'grad_sample' attribute in the given parameter, or accumulate it
    if the 'grad_sample' attribute already exists.
    This custom code will not work when using optimizer.virtual_step()
    """

    if hasattr(param, "grad_sample"):
        param.grad_sample = param.grad_sample + grad_sample
        # param.grad_sample = torch.cat((param.grad_sample, grad_sample), batch_dim)
    else:
        param.grad_sample = grad_sample


class DPCTGAN(CTGANSynthesizer, Synthesizer):
    """DPCTGAN Synthesizer.

    GAN-based synthesizer that uses conditional masks to learn tabular data.

    :param epsilon: Privacy budget for the model.
    :param sigma: The noise scale for the gradients.  Noise scale and batch size influence
        how fast the privacy budget is consumed, which in turn influences the
        convergence rate and the quality of the synthetic data.
    :param batch_size: The batch size for training the model.
    :param epochs: The number of epochs to train the model.
    :param embedding_dim: The dimensionality of the embedding layer.
    :param generator_dim: The dimensionality of the generator layer.
    :param discriminator_dim: The dimensionality of the discriminator layer.
    :param generator_lr: The learning rate for the generator.
    :param discriminator_lr: The learning rate for the discriminator.
    :param verbose: Whether to print the training progress.
    :param diabled_dp: Allows training without differential privacy, to diagnose
        whether any model issues are caused by privacy or are simply the
        result of GAN instability or other issues with hyperparameters.

    """
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
        disabled_dp=False,
        delta=None,
        sigma=5,
        max_per_sample_grad_norm=1.0,
        epsilon=1,
        loss="cross_entropy"
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

        # opacus parameters
        self.sigma = sigma
        self.disabled_dp = disabled_dp
        self.delta = delta
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.epsilon = epsilon
        self.epsilon_list = []
        self.alpha_list = []
        self.loss_d_list = []
        self.loss_g_list = []
        self.verbose = verbose
        self.loss = loss

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

        if self.loss != "cross_entropy":
            # Monkeypatches the _create_or_extend_grad_sample function when calling opacus
            opacus.grad_sample.utils.create_or_extend_grad_sample = (
                _custom_create_or_extend_grad_sample
            )

    def train(
        self,
        data,
        transformer=None,
        categorical_columns=[],
        ordinal_columns=[],
        continuous_columns=[],
        update_epsilon=None, 
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

        train_data = np.array([
            [float(x) if x is not None else 0.0 for x in row] for row in train_data
        ])

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.transformers
        )

        data_dim = self._transformer.output_width

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            self.loss,
            self.pac,
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        privacy_engine = opacus.PrivacyEngine(
            discriminator,
            batch_size=self._batch_size,
            sample_size=train_data.shape[0],
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=self.sigma,
            max_grad_norm=self.max_per_sample_grad_norm,
            clip_per_layer=True,
        )

        if not self.disabled_dp:
            privacy_engine.attach(optimizerD)

        one = torch.tensor(1, dtype=torch.float).to(self._device)
        mone = one * -1

        real_label = 1
        fake_label = 0
        criterion = nn.BCELoss()

        assert self._batch_size % 2 == 0
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(self._epochs):
            if not self.disabled_dp:
                # if self.loss == 'cross_entropy':
                #    autograd_grad_sample.clear_backprops(discriminator)
                # else:
                for p in discriminator.parameters():
                    if hasattr(p, "grad_sample"):
                        del p.grad_sample

                if self.delta is None:
                    self.delta = 1 / (
                        train_data.shape[0] * np.sqrt(train_data.shape[0])
                    )

                epsilon, best_alpha = optimizerD.privacy_engine.get_privacy_spent(
                    self.delta
                )

                self.epsilon_list.append(epsilon)
                self.alpha_list.append(best_alpha)
                if self.epsilon < epsilon:
                    if self._epochs == 1:
                        raise ValueError(
                            "Inputted epsilon and sigma parameters are too small to"
                            + " create a private dataset. Try increasing either parameter "
                            + "and rerunning."
                        )
                    else:
                        break

            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self._data_sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = self._data_sampler.sample_data(self._batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)
                    real = self._data_sampler.sample_data(
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
                    fake_cat = fakeact

                optimizerD.zero_grad()

                if self.loss == "cross_entropy":
                    y_fake = discriminator(fake_cat)

                    #   print ('y_fake is {}'.format(y_fake))
                    label_fake = torch.full(
                        (int(self._batch_size / self.pac),),
                        fake_label,
                        dtype=torch.float,
                        device=self._device,
                    )

                    #    print ('label_fake is {}'.format(label_fake))

                    error_d_fake = criterion(y_fake.squeeze(), label_fake)
                    error_d_fake.backward()
                    optimizerD.step()

                    # train with real
                    label_true = torch.full(
                        (int(self._batch_size / self.pac),),
                        real_label,
                        dtype=torch.float,
                        device=self._device,
                    )
                    y_real = discriminator(real_cat)
                    error_d_real = criterion(y_real.squeeze(), label_true)
                    error_d_real.backward()
                    optimizerD.step()

                    loss_d = error_d_real + error_d_fake

                else:

                    y_fake = discriminator(fake_cat)
                    mean_fake = torch.mean(y_fake)
                    mean_fake.backward(one)

                    y_real = discriminator(real_cat)
                    mean_real = torch.mean(y_real)
                    mean_real.backward(mone)

                    optimizerD.step()

                    loss_d = -(mean_real - mean_fake)

                max_grad_norm = []
                for p in discriminator.parameters():
                    param_norm = p.grad.data.norm(2).item()
                    max_grad_norm.append(param_norm)
                # pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, self.device)

                # pen.backward(retain_graph=True)
                # loss_d.backward()
                # optimizer_d.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

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
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                # if condvec is None:
                cross_entropy = 0
                # else:
                #    cross_entropy = self._cond_loss(fake, c1, m1)

                if self.loss == "cross_entropy":
                    label_g = torch.full(
                        (int(self._batch_size / self.pac),),
                        real_label,
                        dtype=torch.float,
                        device=self._device,
                    )
                    # label_g = torch.full(int(self.batch_size/self.pack,),1,device=self.device)
                    loss_g = criterion(y_fake.squeeze(), label_g)
                    loss_g = loss_g + cross_entropy
                else:
                    loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            self.loss_d_list.append(loss_d)
            self.loss_g_list.append(loss_g)
            if self.verbose:
                print(
                    "Epoch %d, Loss G: %.4f, Loss D: %.4f"
                    % (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()),
                    flush=True,
                )
                print("epsilon is {e}, alpha is {a}".format(e=epsilon, a=best_alpha))

        return self.loss_d_list, self.loss_g_list, self.epsilon_list, self.alpha_list

    def generate(self, n, condition_column=None, condition_value=None):
        """
        TODO: Add condition_column support from CTGAN
        """
        self._generator.eval()

        # output_info = self._transformer.output_info
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            condvec = self._data_sampler.sample_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1, m1, col, opt = condvec
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
