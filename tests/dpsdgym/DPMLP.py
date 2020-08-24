import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pandas as pd

from torchdp import PrivacyEngine, utils, autograd_grad_sample

class DPMLPClassifier(hidden_layer_sizes=(100, ), 
                        activation='relu', 
                        solver='adam', 
                        alpha=0.0001, 
                        batch_size='auto', 
                        learning_rate='constant', 
                        learning_rate_init=0.001, 
                        max_iter=200, 
                        momentum=0.9, 
                        early_stopping=False, 
                        validation_fraction=0.1, 
                        beta_1=0.9, 
                        beta_2=0.999, 
                        epsilon=1e-08, 
                        n_iter_no_change=10, 
                        max_fun=15000):

    class NN(nn.Module):
        def __init__(self, input_size, classes, hidden_layer_sizes):
            super(self.NN, self).__init__()
            self.fc1 = nn.Linear(input_size,hidden_layer_sizes[0])
            self.fc2 = nn.Linear(hidden_layer_sizes[0],hidden_layer_sizes[1])
            self.fc3 = nn.Linear(hidden_layer_sizes[1],classes)
            
        def forward(self, x):
            x = self.fc3(F.leaky_relu(self.fc2(F.leaky_relu(self.fc1(x), 0.2)), 0.2))
            return x
    
    def fit(self, X, y):
        self.net = self.NN(X.shape[1], len(np.unique(y)), (50,20))

        sample_size=len(X)
        batch_size=min(250, len(X))

        optimizer = optim.Adam(self.net.parameters(), lr=.02, betas=(0.5, 0.9))
        criterion = nn.CrossEntropyLoss()

        privacy_engine = PrivacyEngine(
            self.net,
            batch_size,
            sample_size,
            alphas= [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=3.0,
            max_grad_norm=1.0,
            clip_per_layer=True
        )
        privacy_engine.attach(optimizer)

        target_delta = 1/X.shape[0]

        for epoch in range(500):
            for i in range(int(len(X)/batch_size) + 1):
                data2 = X.iloc[i*batch_size:i*batch_size+batch_size, :]
                labels = Y.iloc[i*batch_size:i*batch_size+batch_size, :]
                if len(labels) < batch_size:
                    # TODO: figure out how to avoid this
                    # by using the pytorch dataloader
                    break
                X, Y = Variable(torch.FloatTensor([data2.to_numpy()]), requires_grad=True), Variable(torch.FloatTensor([labels.to_numpy()]), requires_grad=False)
                optimizer.zero_grad()
                y_pred = self.net(X)
                output = criterion(y_pred.squeeze(), Y.squeeze().long())
                output.backward()
                optimizer.step()
                
            if (epoch % 3 == 0.0):
                print("Epoch {} - loss: {}".format(epoch, output))
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta)
                print ('epsilon is {e}, alpha is {a}'.format(e=epsilon, a = best_alpha))
                if 3.0 < epsilon:
                    break
    
    def predict(self, X):
        torch.argmax(self.net(Variable(torch.FloatTensor([X.to_numpy()]), requires_grad=True))[0],1)

    def get_params(self, deep):
        raise NotImplementedError

    def set_params(self, params):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict_log_proba(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError