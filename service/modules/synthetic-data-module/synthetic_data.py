#%%
import mlflow
import json
import sys
import os
import yaml

import numpy as np
import pandas as pd
import math
import random
#%%
from opendp.whitenoise.client import get_dataset_client
from opendp.whitenoise.data.adapters import load_reader, load_metadata, load_dataset
from opendp.whitenoise.sql.private_reader import PrivateReader
from pandasql import sqldf

from sdgym.constants import CONTINUOUS
from sdgym.synthesizers.base import BaseSynthesizer
from sdgym.synthesizers.utils import Transformer

def load_data(dataset_name, budget, benchmark=False):
    """
    Only works with categorical/ordinal columns as of now
    
    SQL scenario...?
    """
    # Load dataset from service (dataset is pd.DataFrame)
    dataset_document = get_dataset_client().read(dataset_name, budget)
    dataset = load_dataset(dataset_document)

    # TODO: 1/10th of the budget on count here

    # Eventually retrieve columns
    # ...

    return dataset.to_numpy()

def parse_metadata(data, categorical_columns=tuple(), ordinal_columns=tuple()):
    meta = []

    df = pd.DataFrame(data)
    for index in df:
        column = df[index]

        if index in categorical_columns:
            mapper = column.value_counts().index.tolist()
            meta.append({
                "name": index,
                "type": CATEGORICAL,
                "size": len(mapper),
                "i2s": mapper
            })
        elif index in ordinal_columns:
            value_count = list(dict(column.value_counts()).items())
            value_count = sorted(value_count, key=lambda x: -x[1])
            mapper = list(map(lambda x: x[0], value_count))
            meta.append({
                "name": index,
                "type": ORDINAL,
                "size": len(mapper),
                "i2s": mapper
            })
        else:
            meta.append({
                "name": index,
                "type": CONTINUOUS,
                "min": column.min(),
                "max": column.max(),
            })

    return meta


class OrdinalSynthesizer(BaseSynthesizer):
    """
    Performs perturbation on synthesized ordinal data.
    Perturbs selection distribution
    """
    def __init__(self):
        self.noise_param = .1

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.dtype = data.dtype
        self.metadata = parse_metadata(data, categorical_columns, ordinal_columns)

        for name, info in enumerate(self.metadata):
            nomial = np.bincount(data[:, name].astype('int'), minlength=info['size'])
            nomial = nomial / np.sum(nomial)
            self.models.append(nomial)

    def sample(self, samples):
        data = np.zeros([samples, len(self.metadata)], self.dtype)
        
        for name, info in enumerate(self.metadata):
            size = len(self.models[name])
            data[:, name] = np.random.choice(np.arange(size), samples, p=self.models[name])
        return data

class MWEMSynthesizer(BaseSynthesizer):
    # Only works for binnable data

    # data is dataframe
    def __init__(self, Bdata, D, Q, T, epsilon, size):
        self.B = B
        self.D = D
        self.Q = Q
        self.T = T
        self.epsilon = epsilon
        self.size = size

# %%
    def Laplace(sigma):
        if random.randint(0,1) == 0:
            return sigma * math.log(random.random()) * -1
        else:
            return sigma * math.log(random.random()) * 1

    def histogram_from_data_attributes(data):
        """
        We need to collect information about the data
        in order to initialize a matrix histogram.
        """
        mins_data = []
        maxs_data = []
        dims_sizes = []
        # Transpose for column wise iteration
        for column in data.T:
            min_c = min(column) ; max_c = max(column) 
            mins_data.append(min_c)
            maxs_data.append(max_c)
            # Dimension size (number of bins)
            dims_sizes.append(max_c-min_c+1)
        
        # Produce an N,D dimensional histogram, where
        # we pre-specify the bin sizes to correspond with 
        # our ranges above
        histogram, _ = np.histogramdd(data, bins=dims_sizes)
        # Return histogram, dimensions
        return histogram, dims_sizes
    
    def initialize_A(histogram, dimensions, eps):
        # TODO: Actually use the distribution from real data with some budget,
        # as opposed to using uniform dist
        A = []
        n = 0
        m = [sum(histogram[i]) for i in range(len(histogram))]
        n = sum(m)
        value = n/np.prod(dimensions)
        A = np.zeros_like(histogram)
        A += value
        return A, eps

    def ExponentialMechanism(B, A, Q, eps):
        errors = [0] * len(Q)
        for i in range(len(errors)):
            errors[i] = eps * abs(Evalu)

    def compose_arbitrary_slices(num_s, dimensions):
        """
        Here, dimensions is the shape of the histogram
        We want to return a list of length num_s, containing
        random slice objects, given the dimensions

        n =[[[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 1.]],
            [[0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.]],
            [[1., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]]]

        Sample dims: (3,3,3)

        """
        slices_list = []
        for _ in range(num_s):
            slices = []
            for i in dims:
                # Random nums within range(0,dim_size)
                a = np.random.randint(i)
                b = np.random.randint(i)
                # Bounds
                l_b = min(a,b) ; u_b = max(a,b)
                # Create a slice for the bounds
                if l_b != u_b:
                    slices.append(np.s_[l_b:u_b])
                    slices.append(np.s_[np.random.randint(l_b,u_b)-l_b])
            slices_list.append(slices)

    def evaluate_slice(data,slices):
        """
        Run through the slices
        """
        for s in slices:
            data = data[s]
        return data

    
    def Evaluate(query, collection):
        #We count the number of "objects" from index x
        #to index y specified by the query = {x,y}
        #e.g: collection = [2, 3, 6, 4, 1], query = {(2,1):(3,3)}
        
        # We want to count the number of objects in an
        # arbitrary slice of our collection

        # We use np.s_[arbitrary slice] as our queries

        key = list(query)[0]
        startInd = min(key[0], key[1])
        endInd = max(key[0], key[1])
        startInd2 = min(query[key][0], query[key][1])
        endInd2 = max(query[key][0], query[key][1])
        counting = 0

        for i in range(startInd, endInd+1):
            for j in range(startInd2, endInd2+1):
                counting += collection[i][j]
        return counting

#%%
n =[[[0., 0., 0.],
  [0., 0., 0.],
  [0., 0., 1.]],
 [[0., 0., 0.],
  [0., 1., 0.],
  [0., 0., 0.]],
 [[1., 0., 0.],
  [0., 0., 0.],
  [0., 0., 0.]]]

# n[0:1]
# n[0:1][0][0:2][1][0:2]
dims = (3,3,3)
slices = []

for i in dims:
    # Random nums within range(0,dim_size)
    a = np.random.randint(i)
    b = np.random.randint(i)
    # Bounds
    l_b = min(a,b) ; u_b = max(a,b)
    # Create a slice for the bounds
    if l_b != u_b:
        slices.append(np.s_[l_b:u_b])
        slices.append(np.s_[np.random.randint(l_b,u_b)-l_b])

n
#%%


df = pd.DataFrame({'num_legs': [2, 4, 8, 0],
                   'num_wings': [2, 0, 0, 0],
                   'num_specimen_seen': [10, 2, 1, 8]},
                  index=['falcon', 'dog', 'spider', 'fish'])
df_simple = pd.DataFrame({'num_legs': [3, 2, 1],
                   'num_seen': [1, 2, 3],
                   'num_wings': [7, 8, 9]},
                  index=['falcon', 'dog', 'turtle'])
nf = df.to_numpy()
nf_simple = df_simple.to_numpy()
hist, dimensions = histogram_from_data_attributes(nf_simple)
epsilon = 3.0
A, epsilon = initialize_A(hist, dimensions, epsilon)

# r = np.random.randn(3,3)
# H, edges = np.histogramdd(r, bins = (3, 3, 3))
# print(H)
#%%
    
    




if __name__ == "__main__":
    private_dataset_name = sys.argv[1]
    release_dataset_name = sys.argv[2]
    budget = sys.argv[3]
    
    with mlflow.start_run():
        data = load_data("iris", budget)
