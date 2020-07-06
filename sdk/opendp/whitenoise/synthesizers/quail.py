import numpy as np
import pandas as pd
import pickle

from opendp.whitenoise.synthesizers.base import SDGYMBaseSynthesizer

class QUAILSynthesizer(SDGYMBaseSynthesizer):
    """
    Qualified Architecture to Improve Labeling.

    Divide epsilon in a known classification task
    between a differentially private synthesizer and
    classifier. Train DP classifier on real, fit DP synthesizer
    to features (excluding the target label)
    and use synthetic data from the DP synthesizer with
    the DP classifier to create artificial labels. Produces
    complete synthetic data
    """
    
    def __init__(self, epsilon, dp_synthesizer, synth_args, dp_classifier, class_args, target, test_size=0.2, seed=42, eps_split=0.9):
        self.epsilon = epsilon
        self.eps_split = eps_split
        self.dp_synthesizer = dp_synthesizer
        self.synth_args = synth_args
        self.dp_classifier = dp_classifier
        self.class_args = class_args
        self.target = target
        self.test_size = test_size
        self.seed = seed
        
        # Model
        self.private_model = None
        self.private_synth = None
        
        # Pandas check
        self.pandas = False
        self.pd_cols = None
        self.pd_index = None
        
    def fit(self, data):
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
            
        private_features = data.loc[:, data.columns != self.target]
        private_target = data.loc[:, data.columns == self.target]
        x_train, x_test, y_train, y_test = train_test_split(private_features, private_target, test_size=self.test_size, random_state=self.seed)
        
        # Here we train a differentially private model on the real
        # data. We report on the accuracy for now to give a sense of
        # the upper bound on performance in the sampling step.
        self.private_model = self.dp_classifier(epsilon=(self.epsilon * self.eps_split), **self.class_args)
        print(y_train)
        self.private_model.fit(x_train, y_train.values.ravel())
        predictions = self.private_model.predict(x_test)
        self.class_report = classification_report(np.ravel(y_test), predictions, labels=np.unique(predictions))
        self.target_accuracy = accuracy_score(np.ravel(y_test), predictions)
        
        print(self.class_report)
        print(self.target_accuracy)

        # We use the features in our synthesis.
        self.private_synth = self.dp_synthesizer(epsilon = (self.epsilon * (1 - self.eps_split)), **self.synth_args)
        self.private_synth.fit(data=private_features)

        if hasattr(self.private_model, 'coef_'):
            print(self.private_model.coef_)
        
        if hasattr(self.private_model, 'intercept_'):
            print(self.private_model.intercept_)

        if hasattr(self.private_model, 'classes_'):
            print(self.private_model.classes_)
        
        model_file = str(self.epsilon) + "_priv_model.pkl"
        with open(model_file, 'wb') as file:
            pickle.dump(self.private_model, file)

    def sample(self, samples):
        sampled_features = self.private_synth.sample(samples)
        Y = self.private_model.predict(sampled_features)

        sampled_features[self.target] = Y
        return sampled_features