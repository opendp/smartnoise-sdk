import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ctgan import CTGANSynthesizer

from .privacy_utils import weights_init, pate, moments_acc

import torch
from torch import optim
from torch.nn import functional
import torch.nn as nn
import torch.utils.data
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,Sigmoid
from torch.nn import functional as F
from opendp.smartnoise.synthesizers.base import SDGYMBaseSynthesizer

import ctgan
from ctgan.transformer import DataTransformer
from ctgan.conditional import ConditionalGenerator

from ctgan.models import Generator

from ctgan.sampler import Sampler

from ctgan import CTGANSynthesizer
from torch.autograd import Variable, grad

class Discriminator(Module):

    def dragan_penalty(self, real_data, device='cpu', c=10, lambda_=10):
        alpha = torch.rand(real_data.shape[0],1, device=device).expand(real_data.shape)
        delta = torch.normal(mean=0.0, std=c, size=real_data.shape, device=device) # 0.5 * real_data.std() * torch.rand(real_data.shape)
        x_hat = Variable(alpha * real_data + (1 - alpha) * (real_data + delta), requires_grad=True)

        pred_hat = self(x_hat.float())

        gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size(), device=device), create_graph=True, retain_graph=True, only_inputs=True)[0]
        dragan_penalty = lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return dragan_penalty

    def __init__(self, input_dim, dis_dims, loss, pack):
        super(Discriminator, self).__init__()
        torch.cuda.manual_seed(0)
        torch.manual_seed(0)

        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        if loss == 'cross_entropy':
            seq += [Sigmoid()]
        self.seq = Sequential(*seq)

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))

class PATECTGAN(CTGANSynthesizer):
    def __init__(self,
                 embedding_dim=128,
                 gen_dim=(256, 256),
                 dis_dim=(256, 256),
                 l2scale=1e-6,
                 epochs=300,
                 pack=1,
                 log_frequency=True,
                 disabled_dp=False,
                 target_delta=None,
                 sigma = 5,
                 max_per_sample_grad_norm=1.0,
                 verbose=False,
                 loss = 'cross_entropy',#losses supported: 'cross_entropy', 'wasserstein'
                 regularization = None, #regularizations supported: 'dragan'
                 binary=False,
                 batch_size = 500,
                 teacher_iters = 5,
                 student_iters = 5,
                 sample_per_teacher = 1000,
                 epsilon = 8.0,
                 delta = 1e-5,
                 noise_multiplier = 1e-3,
                 moments_order = 100):

        # CTGAN model specifi3c parameters
        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.pack=pack
        self.log_frequency = log_frequency
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.verbose=verbose
        self.loss=loss
        self.regularization = regularization if self.loss != 'wasserstein' else 'dragan'
        self.sample_per_teacher = sample_per_teacher
        self.noise_multiplier = noise_multiplier
        self.moments_order = moments_order

        self.binary = binary
        self.batch_size = batch_size
        self.teacher_iters = teacher_iters
        self.student_iters = student_iters
        self.epsilon = epsilon
        self.delta = delta
        self.pd_cols = None
        self.pd_index = None

    def train(self, data, categorical_columns=None, ordinal_columns=None, update_epsilon=None):
        if update_epsilon:
            self.epsilon = update_epsilon

        sample_per_teacher = self.sample_per_teacher if self.sample_per_teacher < len(data) else 1000
        self.num_teachers = int(len(data) / sample_per_teacher) + 1
        self.transformer = DataTransformer()
        self.transformer.fit(data, discrete_columns=categorical_columns)
        data = self.transformer.transform(data)
        data_partitions = np.array_split(data, self.num_teachers)

        data_dim = self.transformer.output_dimensions

        self.cond_generator = ConditionalGenerator(data, self.transformer.output_info, self.log_frequency)

        # create conditional generator for each teacher model
        cond_generator = [ConditionalGenerator(d, self.transformer.output_info, self.log_frequency) for d in data_partitions]

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim).to(self.device)

        discriminator = Discriminator(
            data_dim + self.cond_generator.n_opt,
            self.dis_dim,
            self.loss,
            self.pack).to(self.device)

        student_disc = discriminator
        student_disc.apply(weights_init)

        teacher_disc = [discriminator for i in range(self.num_teachers)]
        for i in range (self.num_teachers):
            teacher_disc[i].apply(weights_init)

        optimizerG = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerS = optim.Adam(student_disc.parameters(), lr=2e-4, betas=(0.5, 0.9))
        optimizerT = [optim.Adam(teacher_disc[i].parameters(), lr=2e-4, betas=(0.5, 0.9)) for i in range(self.num_teachers)]
        
        
        noise_multiplier = self.noise_multiplier
        alphas = torch.tensor([0.0 for i in range(self.moments_order)], device=self.device)
        l_list = 1 + torch.tensor(range(self.moments_order), device=self.device)
        eps = 0

        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        REAL_LABEL = 1
        FAKE_LABEL = 0

        criterion = nn.BCELoss() if(self.loss == "cross_entropy") else self.WLoss

        if(self.verbose):
            print("using loss {} and regularization {}".format(self.loss,self.regularization))

        while eps < self.epsilon:
            # train teacher discriminators
            for t_2 in range(self.teacher_iters):
                for i in range(self.num_teachers):                   
                    partition_data = data_partitions[i]              
                    data_sampler = Sampler(partition_data, self.transformer.output_info)
                    fakez = torch.normal(mean, std=std).to(self.device)

                    condvec = cond_generator[i].sample(self.batch_size)

                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = data_sampler.sample(self.batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self.device)
                        m1 = torch.from_numpy(m1).to(self.device)
                        fakez = torch.cat([fakez, c1], dim=1)
                        perm = np.arange(self.batch_size)
                        np.random.shuffle(perm)
                        real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self.generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self.device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fake

                    optimizerT[i].zero_grad()

                    y_all = torch.cat([teacher_disc[i](fake_cat), teacher_disc[i](real_cat)])
                    label_fake = torch.full((int(self.batch_size/self.pack),1), FAKE_LABEL, dtype=torch.float, device=self.device)
                    label_true = torch.full((int(self.batch_size/self.pack),1), REAL_LABEL, dtype=torch.float, device=self.device)
                    labels = torch.cat([label_fake, label_true])
                    
                    errD = criterion(y_all, labels)
                    errD.backward()

                    if(self.regularization == 'dragan'):
                        pen = teacher_disc[i].dragan_penalty(real_cat, device=self.device)
                        pen.backward(retain_graph=True)
                    
                    optimizerT[i].step()

            # train student discriminator
            for t_3 in range(self.student_iters):
                data_sampler = Sampler(data, self.transformer.output_info)
                fakez = torch.normal(mean, std=std)

                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                else:
                    fake_cat = fake

                fake_data = fake_cat
                predictions, votes = pate(fake_data, teacher_disc, noise_multiplier, device=self.device)

                output = student_disc(fake_data.detach())

                # update moments accountant
                alphas = alphas + moments_acc(self.num_teachers, votes, noise_multiplier, l_list, device=self.device)

                loss_s = criterion(output, predictions.float().to(self.device))

                optimizerS.zero_grad()
                loss_s.backward()

                if(self.regularization == 'dragan'):
                    vals = torch.cat([predictions, fake_data], axis=1)
                    ordered = vals[vals[:,0].sort()[1]]
                    data_list = torch.split(ordered, predictions.shape[0] - int(predictions.sum().item()))
                    synth_cat = torch.cat(data_list[1:], axis=0)[:, 1:]
                    pen = student_disc.dragan_penalty(synth_cat, device=self.device)
                    pen.backward(retain_graph=True)

                optimizerS.step()

            # print ('iterator {i}, student discriminator loss is {j}'.format(i=t_3, j=loss_s))

            #train generator
            fakez = torch.normal(mean=mean, std=std)
            condvec = self.cond_generator.sample(self.batch_size)

            if condvec is None:
                c1, m1, col, opt = None, None, None, None
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                m1 = torch.from_numpy(m1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)
    
            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
    
            if c1 is not None:
                y_fake = student_disc(torch.cat([fakeact, c1], dim=1))
            else:
                y_fake = student_disc(fakeact)
    
            if condvec is None:
                cross_entropy = 0
            else:
                cross_entropy = self._cond_loss(fake, c1, m1)

            if self.loss=='cross_entropy':
               label_g = torch.full((int(self.batch_size/self.pack),1), REAL_LABEL, dtype=torch.float, device=self.device)
               loss_g = criterion(y_fake, label_g.float())
               loss_g = loss_g + cross_entropy
            else:
               loss_g = -torch.mean(y_fake) + cross_entropy

            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()

            eps = min((alphas - math.log(self.delta)) / l_list)
            
            if(self.verbose):
                print ('eps: {:f} \t G: {:f} \t D: {:f}'.format(eps, loss_g.detach().cpu(), loss_s.detach().cpu()))

    def WLoss(self, output, labels):
        vals = torch.cat([labels, output], axis=1)
        ordered = vals[vals[:, 0].sort()[1]]
        data_list = torch.split(ordered, labels.shape[0] - int(labels.sum().item()))
        fake_score = data_list[0][:, 1]
        true_score = torch.cat(data_list[1:], axis=0)[:, 1]
        w_loss = -(torch.mean(true_score) - torch.mean(fake_score))
        return w_loss

    def generate(self, n):
        self.generator.eval()

        steps = n // self.batch_size + 1

        data = []

        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.cond_generator.sample_zero(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        generated_data = self.transformer.inverse_transform(data, None)

        return generated_data
