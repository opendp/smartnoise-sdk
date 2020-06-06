import numpy as np
import torch
from torch import optim
from torch.nn import functional
import torch.nn as nn

from dpctgan.conditional import ConditionalGenerator
from dpctgan.models import Discriminator, Generator
from dpctgan.sampler import Sampler
from dpctgan.transformer import DataTransformer
from torchdp import PrivacyEngine, utils
from torchdp import autograd_grad_sample


class DPCTGANSynthesizer(object):
    """Conditional Table GAN Synthesizer.
    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        gen_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Resiudal Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        dis_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        l2scale (float):
            Wheight Decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
    """

    def __init__(self, embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256),
                 l2scale=1e-6, batch_size=500,disabled_dp=True, target_delta=None,sigma=0,max_per_sample_grad_norm=1.0):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sigma = sigma
        self.disabled_dp = disabled_dp
        self.target_delta = target_delta
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.epsilon_list = []
        self.alpha_list = []
     #   torch.cuda.manual_seed(0)
     #   torch.manual_seed(0)

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == 'softmax':
                ed = st + item[0]
                data_t.append(functional.gumbel_softmax(data[:, st:ed], tau=0.2))
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        skip = False
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True

            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                ed_c = st_c + item[0]
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction='none'
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

            else:
                assert 0

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def fit(self, train_data, discrete_columns=tuple(), epochs=300, pack=10, log_frequency=True, loss='cross_entropy'):
        """Fit the CTGAN Synthesizer models to the training data.
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a
                pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            epochs (int):
                Number of training epochs. Defaults to 300.
            log_frequency (boolean):
                Whether to use log frequency of categorical levels in conditional
                sampling. Defaults to ``True``.
        """

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        
        one = torch.tensor(1, dtype=torch.float).to(self.device)
        mone= one * -1

        data_sampler = Sampler(train_data, self.transformer.output_info)
        
       

        data_dim = self.transformer.output_dimensions
        self.cond_generator = ConditionalGenerator(
            train_data,
            self.transformer.output_info,
            log_frequency
        )

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim
        ).to(self.device)
        
        
     #   print ('first input to disciminator is {}'.format(data_dim + self.cond_generator.n_opt))

        discriminator = Discriminator(
            data_dim + self.cond_generator.n_opt,
            self.dis_dim,
            pack
        ).to(self.device)
        
    #    print (data_dim + self.cond_generator.n_opt)
    #    print (self.dis_dim)
        
        print (list(discriminator.modules()))
#
        optimizerG = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9),
            weight_decay=self.l2scale
        )
    
        REAL_LABEL = 1
        FAKE_LABEL = 0
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        #optimizerD = optim.SGD(discriminator.parameters(), lr=2e-4)
        
   #     print ('discrminator model parameters')
   #     print (list(discriminator.parameters()))
    
        criterion = nn.BCELoss()
        
        
        privacy_engine = PrivacyEngine(
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

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):
          #      print ('step {} in this epoch'.format(id_))
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.sample(self.batch_size)
               # condvec = None
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
                    
            #    print ('fakez size is {}'.format(fakez.shape))

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)
                
                
            #    print ('real data is {}'.format(real.shape))

                real = torch.from_numpy(real.astype('float32')).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake
                    
                optimizerD.zero_grad()
            #    pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, self.device)
            #    pen.backward(retain_graph=True)
            
             #   print ('fake cat shape is {}'.format(fake_cat.shape))
             #   print ('check_discriminator again')
             #   print (list(discriminator.parameters()))
                if loss == 'cross_entropy':
                    y_fake = discriminator(fake_cat)
                   # loss_d_fake = nn.BCELoss()
                    
                    # train with fake
                    label_fake = torch.full((int(self.batch_size/pack),), FAKE_LABEL, device=self.device)
                    #output = netD(fake.detach())
                    errD_fake = criterion(y_fake, label_fake)
                 #   print ('erroD_fake is {}'.format(errD_fake))
                    errD_fake.backward()
                    optimizerD.step()

                    # train with real
                    label_true = torch.full((int(self.batch_size/pack),), REAL_LABEL, device=self.device)
                    y_real = discriminator(real_cat)
                    errD_real = criterion(y_real, label_true)
                 #   print ('errD_real is {}'.format(errD_real))
                    errD_real.backward()
                    optimizerD.step()
                    #D_x = output.mean().item()

                    #D_G_z1 = output.mean().item()
                    loss_d = errD_real + errD_fake
                else:
                    y_fake = discriminator(fake_cat)
                
                    loss_d_fake = torch.mean(y_fake)
                    loss_d_fake.backward(one)
                    #optimizerD.step()
                
           #     print ('real cat is ')
           #     print ('real cat shape is {}'.format(real_cat.shape))
             #   optimizerD.zero_grad()
                    y_real = discriminator(real_cat)
                    loss_d_real = torch.mean(y_real)
                    loss_d_real.backward(mone)
                    optimizerD.step()
               # loss_d_real = torch.mean(y_real)
               # loss_d_real.backward(mone)
               # optimizerD.step()
            #    

                
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
              #  loss_d = (-loss_d_real - loss_d_fake)
                   # loss_d.backward()
                    #print ('this should not happen')
                    #optimizerD.step()
                    
                max_grad_norm = []
                for p in discriminator.parameters():
                    param_norm = p.grad.data.norm(2).item()
                    max_grad_norm.append(param_norm)
              #  d

                #if self.disabled_dp:
                    
                   # print (pen)
               # pen.backward(retain_graph=True)
              #  loss_d.backward()
              #  optimizerD.step()
            #    print ("check grad")
             #   for layer in discriminator.modules():
              #      try:
              #          print (layer.weight.grad)
               #     except:
                #        print (layer)
             #   if not self.disabled_dp:       
                    #optimizerD.virtual_step()
              #      optimizerD.step()
              #  else:
              #      optimizerD.step()
              #  
                #after grad 
            #    print ('after step check param')
             #   print (list(discriminator.parameters()))
                       
                       

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
                
              #  print ('fakeact2 shape is {}'.format(fakeact.shape))
            
                optimizerG.zero_grad()

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)
                    
                # need del param.grad_sample 

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)
                    
                if loss == 'cross_entropy':
                    label_g = torch.full(int(self.batch_size/pack,),1,device=self.device)
                    loss_g = criterion(y_fake, label_g)
                    loss_g = loss_g + cross_entropy
                else:
                    loss_g = -torch.mean(y_fake) + cross_entropy

                
                loss_g.backward()
                optimizerG.step()
                
                
                if not self.disabled_dp:
                    for p in discriminator.parameters():
                        if hasattr(p, "grad_sample"):
                            del p.grad_sample
                            
                #    check = [p.grad_sample for p in discriminator.parameters() if hasattr(p, "grad_sample")]
                #    print ('check if there is any p.gramsample later {}'.format(len(check)))
                    
                  #  autograd_grad_sample.clear_backprops(discriminator)
                    if self.target_delta is None:
                        self.target_delta = 1/train_data.shape[0]
                    epsilon, best_alpha = optimizerD.privacy_engine.get_privacy_spent(self.target_delta)
                    self.epsilon_list.append(epsilon)
                    self.alpha_list.append(best_alpha)

            print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
                  (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()),
                  flush=True)

    def sample(self, n):
        """Sample data similar to the training data.
        Args:
            n (int):
                Number of rows to sample.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """

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

