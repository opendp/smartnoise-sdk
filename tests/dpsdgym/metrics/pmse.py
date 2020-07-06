import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def pmse_ratio(data, synthetic_data):
    """
    To compare the overall similarity of the joint distribution 
    and to calculate the general quality of the synthetic copies
    we rely on the propensity score mean squared error (pMSE) 
    ratio score for synthetic data as proposed by Snoke et al.
    (2018). The pMSE requires training a discriminator 
    that is capable to distinguish between real
    and synthetic examples. A synthetic data set has high 
    general data quality, if the model cannot distinguish between
    real and fake examples.
    From "Really Useful Synthetic Data
    A Framework To Evaluate The Quality Of
    Differentially Private Synthetic Data"
    https://arxiv.org/pdf/2004.07740.pdf
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