import sys
import os
import math
import random

import numpy as np
import pandas as pd

class MWEMSynthesizer():
    """
    N-Dimensional numpy implementation of MWEM. 
    (http://users.cms.caltech.edu/~katrina/papers/mwem-nips.pdf)

    From the paper:
    "[MWEM is] a broadly applicable, simple, and easy-to-implement algorithm, capable of
    substantially improving the performance of linear queries on many realistic datasets...
    (circa 2012)...MWEM matches the best known and nearly
    optimal theoretical accuracy guarantees for differentially private 
    data analysis with linear queries."

    Linear queries used for sampling in this implementation are
    random contiguous slices of the n-dimensional numpy array. 
    """
    def __init__(self, Q_count=400, epsilon=3.0, iterations=150, mult_weights_iterations=20):
        # TODO: Perform check that data is ndarray
        self.Q_count = Q_count
        self.epsilon = epsilon
        self.iterations = iterations
        self.mult_weights_iterations = mult_weights_iterations
        self.synthetic_data = None
        self.data_bins = None
        self.real_data = None

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        """
        Creates a synthetic histogram distribution, based on the original data
        """
        self.data = data.copy()
        self.histogram, self.dimensions, self.data_bins = self.histogram_from_data_attributes(self.data)
        self.Q = self.compose_arbitrary_slices(self.Q_count, self.dimensions)
        # TODO: Add special support for categorical+ordinal columns

        # Run the algorithm
        self.synthetic_data, self.real_data = self.mwem()

    def sample(self, samples):
        """
        Samples from the synthetic histogram
        """
        fake = self.synthetic_data

        s = []
        fake_indices = np.arange(len(np.ravel(fake)))
        fake_distribution = np.ravel(fake)
        norm = np.sum(fake)

        for _ in range(samples):
            s.append(np.random.choice(fake_indices, p=(fake_distribution/norm)))

        s_unraveled = []
        for ind in s:
            s_unraveled.append(np.unravel_index(ind,fake.shape))

        return s_unraveled

    def mwem(self):
        A, epsilon = self.initialize_A(self.histogram, self.dimensions, self.epsilon)
        measurements = {}

        for i in range(self.iterations):
            print("Iteration: " + str(i))

            qi = self.exponential_mechanism(self.histogram, A, self.Q, (epsilon / (2*self.iterations)))

            # Make sure we get a different query to measure:
            while(qi in measurements):
                qi = self.exponential_mechanism(self.histogram, A, self.Q, (epsilon / (2*self.iterations)))

            # NOTE: Add laplace noise here with budget
            evals = self.evaluate(self.Q[qi], self.histogram)
            lap = self.laplace((2*self.iterations)/(self.epsilon*len(self.dimensions)))
            measurements[qi] = evals + lap

            # Improve approximation with Multiplicative Weights
            A = self.multiplicative_weights(A, self.Q, measurements, self.histogram, self.mult_weights_iterations)

        return A, self.histogram
    
    def initialize_A(self, histogram, dimensions, eps):
        # NOTE: Could actually use a distribution from real data with some budget,
        # as opposed to using this uniform dist
        n = np.sum(histogram)
        value = n/np.prod(dimensions)
        A = np.zeros_like(histogram)
        A += value
        return A, eps

    def histogram_from_data_attributes(self, data):
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
        histogram, bins = np.histogramdd(data, bins=dims_sizes)
        # Return histogram, dimensions
        return histogram, dims_sizes, bins
    
    def exponential_mechanism(self, hist, A, Q, eps):
        """
        "Sample a query qi in Q using the Exponential Mechanism
        parametrized with epsilon value epsilon/2T and the score function"
        """
        errors = np.zeros(len(Q))

        for i in range(len(errors)):
            errors[i] = eps * abs(self.evaluate(Q[i], hist)-self.evaluate(Q[i], A))/2.0

        maxi = max(errors)

        for i in range(len(errors)):
            errors[i] = math.exp(errors[i] - maxi)

        uni = np.sum(errors) * random.random()

        for i in range(len(errors)):
            uni -= errors[i]

            if uni <= 0.0:
                return i

        return len(errors) - 1
    
    def multiplicative_weights(self, A, Q, m, hist, iterate):
        sum_A = np.sum(A)

        for _ in range(iterate):
            for qi in m:
                error = m[qi] - self.evaluate(Q[qi], A)

                # Perform the weights update
                query_update = self.binary_replace_in_place_slice(np.zeros_like(A.copy()), Q[qi])
                
                # Apply the update
                A_multiplier = np.exp(query_update * error/(2.0 * sum_A))
                A_multiplier[A_multiplier == 0.0] = 1.0
                A = A * A_multiplier

                # Normalize again
                count_A = np.sum(A)
                A = A * (sum_A/count_A)
        return A
    
    def evaluate(self, a_slice, data):
        """
        Evaluate a count query i.e. an arbitrary slice
        """
        # We want to count the number of objects in an
        # arbitrary slice of our collection

        # We use np.s_[arbitrary slice] as our queries
        e = data.flatten()[a_slice]
        
        if isinstance(e, np.ndarray):
            return np.sum(e)
        else:
            return e

    def compose_arbitrary_slices(self, num_s, dimensions):
        """
        Here, dimensions is the shape of the histogram
        We want to return a list of length num_s, containing
        random slice objects, given the dimensions

        These are our linear queries
        """
        slices_list = []
        # TODO: For analysis, generate a distribution of slice sizes,
        # by running the list of slices on a dimensional array
        # and plotting the bucket size
        for _ in range(num_s):
            # Random linear sample, within dimensions
            # i.e. a contiguous query for the flattened dims
            len_ind = np.prod(dimensions)
            a = np.random.randint(len_ind)
            b = np.random.randint(len_ind)
            while a == b:
                a = np.random.randint(len_ind)
                b = np.random.randint(len_ind)
            # Set bounds and add the slice
            l_b = min(a,b) ; u_b = max(a,b)
            slices_list.append(np.s_[l_b:u_b])
        return slices_list

    def binary_replace_in_place_slice(self, data, a_slice):
        """
        We want to create a binary copy of the data,
        so that we can easily perform our error multiplication
        in MW
        """
        view = data.flatten()

        view[a_slice] = 1.0

        # Recreate the shape of the flattened data
        dim_arr_offset = np.prod(data.shape)
        return view[0:dim_arr_offset].reshape(data.shape)
    
    def laplace(self, sigma):
        return sigma * np.log(random.random()) * np.random.choice([-1, 1])

