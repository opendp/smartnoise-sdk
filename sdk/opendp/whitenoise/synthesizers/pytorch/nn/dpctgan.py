import numpy as np
import torch
from torch import optim
from torch.nn import functional
import torch.nn as nn
import torch.utils.data
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,Sigmoid
from torch.nn import functional as F
from opendp.whitenoise.synthesizers.base import SDGYMBaseSynthesizer

import ctgan
from ctgan.transformer import DataTransformer
from ctgan.conditional import ConditionalGenerator 
from ctgan.models import Discriminator, Generator
from ctgan.sampler import Sampler 

from ctgan import CTGANSynthesizer

import torchdp
from torchdp import autograd_grad_sample
from torchdp import PrivacyEngine, utils



def __init__(self, input_dim, dis_dims, loss, pack):
        super(Discriminator, self).__init__()
        torch.cuda.manual_seed(0)
        torch.manual_seed(0)
        
        dim = input_dim * pack
      #  print ('now dim is {}'.format(dim))
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        if loss == 'cross_entropy':
            print ("this should happen")
            seq += [Sigmoid()]
        self.seq = Sequential(*seq)
        
Discriminator.__init__ = __init__







class DPCTGAN(SDGYMBaseSynthesizer, CTGANSynthesizer):
    """Differential Private Conditional Table GAN Synthesizer
    This code adds Differential Privacy to CTGANSynthesizer from https://github.com/sdv-dev/CTGAN 
    """

    def __init__(self,
                 embedding_dim=128,
                 gen_dim=(256, 256),
                 dis_dim=(256, 256),
                 l2scale=1e-6,
                 batch_size=500,
                 epochs=300,
                 pack=1,
                 log_frequency=True,
                 disabled_dp=False,
                 target_delta=None,
                 sigma = 5,
                 max_per_sample_grad_norm=1.0,
                 budget = 3,
                 verbose=True,
                 loss = 'cross_entropy'

                 ):

        # CTGAN model specific parameters
        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.pack=pack
        self.log_frequency = log_frequency
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # torchdp parameters
        self.sigma = sigma
        self.disabled_dp = disabled_dp
        self.target_delta = target_delta
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.budget = budget
        self.epsilon_list = []
        self.alpha_list = []
        self.loss_d_list = []
        self.loss_g_list = []
        self.verbose=verbose
        self.loss=loss


    def train(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):

        
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, categorical_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dimensions
        self.cond_generator = ConditionalGenerator(train_data, self.transformer.output_info, self.log_frequency)

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim).to(self.device)

        discriminator = Discriminator(
            data_dim + self.cond_generator.n_opt,
            self.dis_dim, 
            self.loss,
            self.pack).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        
        privacy_engine = torchdp.PrivacyEngine(
            discriminator,
            batch_size=self.batch_size,
            sample_size=train_data.shape[0],
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=self.sigma,
            max_grad_norm=self.max_per_sample_grad_norm,
            clip_per_layer=True
        )

        if not self.disabled_dp:
            privacy_engine.attach(optimizerD)

        one = torch.tensor(1, dtype=torch.float).to(self.device)
        mone = one * -1

        REAL_LABEL = 1 
        FAKE_LABEL = 0
        criterion = nn.BCELoss()



        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = len(train_data) // self.batch_size
        for i in range(self.epochs):
            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

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

                real = torch.from_numpy(real.astype('float32')).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                optimizerD.zero_grad()


                if self.loss == 'cross_entropy':
                    y_fake = discriminator(fake_cat)
                    
                 #   print ('y_fake is {}'.format(y_fake))
                    label_fake = torch.full((int(self.batch_size/self.pack),), FAKE_LABEL, device=self.device)
                    
                #    print ('label_fake is {}'.format(label_fake))
   
                    errD_fake = criterion(y_fake, label_fake)
                    errD_fake.backward()
                    optimizerD.step()

                    # train with real
                    label_true = torch.full((int(self.batch_size/self.pack),), REAL_LABEL, device=self.device)
                    y_real = discriminator(real_cat)
                    errD_real = criterion(y_real, label_true)
                    errD_real.backward()
                    optimizerD.step()

                    loss_d = errD_real + errD_fake

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
                #pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, self.device)

                
                #pen.backward(retain_graph=True)
                #loss_d.backward()
                #optimizerD.step()

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
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                if self.loss=='cross_entropy':
                    label_g = torch.full((int(self.batch_size/self.pack),), REAL_LABEL, device=self.device)
                    #label_g = torch.full(int(self.batch_size/self.pack,),1,device=self.device)
                    loss_g = criterion(y_fake, label_g)
                    loss_g = loss_g + cross_entropy
                else:
                    loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

                if not self.disabled_dp:
                    if self.loss == 'cross_entropy':
                        autograd_grad_sample.clear_backprops(discriminator)
                    else:
                        for p in discriminator.parameters():
                            if hasattr(p, "grad_sample"):
                                del p.grad_sample 

                    if self.target_delta is None:
                        self.target_delta = 1/train_data.shape[0]

                    epsilon, best_alpha = optimizerD.privacy_engine.get_privacy_spent(self.target_delta)
                    
                    self.epsilon_list.append(epsilon)
                    self.alpha_list.append(best_alpha)
                    #if self.verbose:
                        
            
            if not self.disabled_dp:
                if self.budget < epsilon:
                    break
            self.loss_d_list.append(loss_d)
            self.loss_g_list.append(loss_g)
            if self.verbose:
                print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
                  (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()),
                  flush=True)
                print ('epsilon is {e}, alpha is {a}'.format(e=epsilon, a = best_alpha))

        return self.loss_d_list, self.loss_g_list, self.epsion_list, self.alpha_list


    def generate(self, n):
        self.generator.eval()

        #output_info = self.transformer.output_info
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
        return self.transformer.inverse_transform(data, None)