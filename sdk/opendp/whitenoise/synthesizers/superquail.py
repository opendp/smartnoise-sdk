import numpy as np
import pandas as pd
import random
import time
import pickle
from joblib import Parallel, delayed

from opendp.whitenoise.synthesizers.base import SDGYMBaseSynthesizer

class dumb_predictor():
    """
    Dummy classifier to be used if any of conf.KNOWN_MODELS break.
    Returns single class as prediction.
    """
    def __init__(self, label):
        self.label = label
        
    def predict(self, instances):
        return np.full(len(instances), self.label)

class SuperQUAILSynthesizer(SDGYMBaseSynthesizer):
    """
    Quailified Architecture to Improve Labeling.

    Divide epsilon in a known classification task
    between a differentially private synthesizer and
    classifier. Train DP classifier on real, fit DP synthesizer
    to features (excluding the target label)
    and use synthetic data from the DP synthesizer with
    the DP classifier to create artificial labels. Produces
    complete synthetic data
    """
    
    def __init__(self, epsilon, dp_classifier, class_args, test_size=0.2, seed=42, eps_split=0.9):
        self.epsilon = epsilon
        self.eps_split = eps_split
        self.dp_classifier = dp_classifier
        self.class_args = class_args
        self.test_size = test_size
        self.seed = seed
        
        # Model
        self.private_models = None
        self.private_synth = None
        
        # Pandas check
        self.pandas = False
        self.pd_cols = None
        self.pd_index = None

        self.continuous_ranges = None
        self.categorical_ranges = None
        self.ordinal_ranges = None
        
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score
        
        if isinstance(data, pd.DataFrame):
            self.pandas = True
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='ignore')
            self.data = data
            self.pd_cols = data.columns
            self.pd_index = data.index
        else:
            raise('Only pandas dataframes for data as of now.')

        self.private_models = {}
        eps_per_column = self.epsilon / float(len(data.columns) + 1)
        
        self.continuous_ranges = {}
        self.categorical_ranges = {}
        self.ordinal_ranges = {}

        for c in categorical_columns:
            # TODO: Delve into this
            # Pretty sure its safe to just grab all the possible categories
            print(data[c])
            self.categorical_ranges[c] = np.unique(data[c])

        for c in ordinal_columns:
            # We do same thing we do for ordinal
            self.ordinal_ranges[c] = (int(self._report_noisy_max_min(data[c], eps_per_column, 'min')),
                                        int(self._report_noisy_max_min(data[c], eps_per_column, 'max')))

        for c in data.columns:
            ## Take care of continuous column distribution ranges here
            # print('Training model for ' +  c)
            if c not in list(categorical_columns) + list(ordinal_columns):
                self.continuous_ranges[c] = (self._report_noisy_max_min(data[c], eps_per_column, 'min'),
                                        self._report_noisy_max_min(data[c], eps_per_column, 'max'))

            ## Train Model
            features = data.loc[:, data.columns != c]
            target = data.loc[:, data.columns == c]
            x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=self.test_size, random_state=self.seed)
            try:
                private_model = self.dp_classifier(epsilon=eps_per_column, **self.class_args)
                private_model.fit(x_train, y_train.values.ravel())
#                 predictions = self.private_model.predict(x_test)
#                 self.class_report = classification_report(np.ravel(y_test), predictions, labels=np.unique(predictions))
#                 self.target_accuracy = accuracy_score(np.ravel(y_test), predictions)
            except:
                print('Unsuccessful when training model for ' +  c + ', using dumb_predictor.')
                y, counts = np.unique(y_train.values.ravel(), return_counts=True)
                label = y[np.argmax(counts)]
                private_model = dumb_predictor(label)
                
            
            if c not in self.private_models:
                self.private_models[c] = private_model
            else:
                raise ValueError("Duplicate column model built.")

    def sample(self, samples):

        def _a_sample(arg):
            index, sample_shape = arg
            sample = np.empty(sample_shape)
            shuffled_column_indexes = np.arange(len(self.data.columns))
            np.random.shuffle(shuffled_column_indexes)
            
            shuffled_columns = self.data.columns[shuffled_column_indexes]
            reordered = self._reorder(shuffled_column_indexes)
            for i, c in enumerate(shuffled_columns):
                pred_sample = np.empty(sample_shape)
                pred_sample[:] = None
                for j, col in enumerate(shuffled_columns):
                    if c != col:
                        if sample[j]:
                            pred_sample[j] = sample[j]
                        else:
                            if col in self.continuous_ranges:
                                pred_sample[j] = np.random.uniform(self.continuous_ranges[col][0],
                                                                    self.continuous_ranges[col][0],1)
                            elif col in self.ordinal_ranges:
                                pred_sample[j] = np.random.randint(self.ordinal_ranges[col][0],
                                                                    self.ordinal_ranges[col][1] + 1)
                            elif col in self.categorical_ranges:
                                pred_sample[j] = np.random.choice(self.categorical_ranges[col],1)
                pred_sample = pred_sample[pred_sample != np.array(None)]
                pred_sample = pred_sample[~np.isnan(pred_sample)]
                # print(c)
                # print(pred_sample)
                c_pred = self.private_models[c].predict(pred_sample.reshape(1, -1))
                sample[i] = c_pred
            return sample[reordered]

        start = time.time()
        job_num = 10
        sample_shape = self.data.iloc[0].shape

        runs = [(i, sample_shape) for i in range(samples)]
        #_a_sample(runs[0])
        results = Parallel(n_jobs=job_num, verbose=1, backend="loky")(
            map(delayed(_a_sample), runs))
        end = time.time() - start
        # print('Sampling took ' + str(end))

        return pd.DataFrame(np.array(results), columns = self.data.columns)
        

    def _report_noisy_max_min(self, set, epsilon, min_or_max='max'):
        best = 0
        r = 0
        for i,d in enumerate(set):
            d = d + self._laplace(epsilon)
            if min_or_max == 'min':
                if d < best:
                    r = i
            elif min_or_max == 'max':
                if d > best:
                    r = i
            else:
                raise ValueError('Must specify either min or max.')
        return r
    
    def _laplace(self, sigma):
        """
        Laplace mechanism

        :param sigma: Laplace scale param sigma
        :type sigma: float
        :return: Random value from laplace distribution [-1,1]
        :rtype: float
        """
        return sigma * np.log(random.random()) * np.random.choice([-1, 1])
    
    def _reorder(self, splits):
        """
        Given an array of dimensionality splits (column indices)
        returns the corresponding reorder array (indices to return
        columns to original order)

        Example:
        original = [[1, 2, 3, 4, 5, 6],
        [ 6,  7,  8,  9, 10, 11]]
        
        splits = [[1,3,4],[0,2,5]]
        
        mod_data = [[2 4 5 1 3 6]
                [ 7  9 10  6  8 11]]
        
        reorder = [3 0 4 1 2 5]

        :param splits: 2d list with splits (column indices)
        :type splits: array of arrays
        :return: 2d list with splits (column indices)
        :rtype: array of arrays
        """
        flat = splits.ravel()
        reordered = np.zeros(len(flat))
        for i, ind in enumerate(flat):
            reordered[ind] = i
        return reordered.astype(int)