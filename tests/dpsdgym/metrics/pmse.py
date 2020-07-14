import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def pmse_ratio(data, synthetic_data):
    """
    In order to determine how similar the synthetic and real data are
    to each other (general quality of synthetic) we can train a 
    discriminator to attempt to distinguish between real and
    synthetic. The poorer the performance of the discriminator, the
    more similar the two datasets are.

    From "Really Useful Synthetic Data
    A Framework To Evaluate The Quality Of
    Differentially Private Synthetic Data"
    https://arxiv.org/pdf/2004.07740.pdf

    :param data: Original data
    :type data: pandas DataFrame
    :param synthetic_data: Synthetic data we are analyzing
    :type synthetic_data: pandas DataFrame
    :return: ratio (pmse score)
    :rtype: float
    """
    n1 = data.shape[0]
    n2 = synthetic_data.shape[0]
    comb = pd.concat([data, synthetic_data], axis=0, keys=[0, 1]).reset_index(level=[0]).rename(columns={'level_0': 'indicator'})
    X_comb = comb.drop('indicator', axis=1)
    y_comb = comb['indicator']
    X_train, X_test, y_train, y_test = train_test_split(X_comb, y_comb, test_size=0.33, random_state=42)
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    score = clf.predict_proba(X_comb)[:, 1]
    observed_utility = sum((score - n2/(n1+n2))**2) / (n1+n2)
    expected_utility = clf.coef_.shape[1] * (n1/(n1+n2))**2 * (n2/(n1+n2)) / (n1+n2)
    ratio = observed_utility / expected_utility
    return ratio