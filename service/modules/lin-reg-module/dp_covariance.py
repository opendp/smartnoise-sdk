import pandas as pd
import numpy as np

class DPcovariance():

    # TODO: check out https://github.com/privacytoolsproject/PSI-Library/blob/2b71facd614845e548be9459ca94bc979eed0d4f/R/statistic-covariance.R#L101
    #   & add properties & etc

    def __init__(self, n, m, rng, epsilon=None, globalEps=None, epsilonDist= None,
                        accuracy=None, accuracyVals=None, imputeRng=None,
                        alpha=0.05):
        self.num_rows = n
        self.num_cols = m

        # TODO set epsilons & sensitivity as necessary


    def release(self, data):
        # TODO: check out the following links to help with implementation
        #  https://github.com/privacytoolsproject/PSI-Library/blob/2b71facd614845e548be9459ca94bc979eed0d4f/R/mechanism-laplace.R#L57
        #  https://github.com/privacytoolsproject/PSI-Library/blob/442dcd451ee5e99229b8df11fddae406cd95aaf5/R/utilities-data-validation.R#L218
        #  https://github.com/privacytoolsproject/PSI-Library/blob/442dcd451ee5e99229b8df11fddae406cd95aaf5/R/utilities-data-validation.R
        return None

    # TODO: this implementation only works for one dependent variable right now
    def get_linear_regression(self, data, x_names, y_names, intercept=False):
        """
        Takes in data, lists of feature names and target names, and whether or not
        we should calculate a y-intercept; and returns a DP linear regression model

        Args:
            data (Dataframe): the data that will be used to make the linear regression
            x (list): list of names of the features (i.e. independent variables) to use
            y (string): name of the target (i.e. dependent variable) to use
            intercept (boolean): true if the lin-reg equation should have a y-intercept, false if not
        Return:
            a list of the coefficients for the linear model
        """
        released_covar_values = self.release(data)
        formatted_covar = make_covar_symmetric(released_covar_values, x_names + y_names)
        lin_reg = cov_method_lin_reg(formatted_covar, self.num_rows, x_names, y_names, intercept)
        # TODO convert lin_reg from Dataframe to sklearn somehow?
        return list(lin_reg['Estimate'])


def make_covar_symmetric(covar, columns):
    """
    Converts unique private covariances into symmetric matrix

    Args:
        covar (???): differentially privately release of elements in lower triangle of covariance matrix
        columns (list): a list of columns (x & y) to be included in the output
    Returns:
        A symmetric differentially private covariance matrix (Dataframe)
    """
    return None


def cov_method_lin_reg(release, num_rows, x_names, y_name, intercept=False):
    """
        Takes in a differentially privately released covariance matrix, the number of rows in the
        original data, whether or not a y-intercept should be calculated, a list of
        feature names, and a target name; and returns a DP linear regression model

        Args:
            release (Dataframe): differentially privately released covariance matrix that will be used to make the linear regression
            num_rows (int): the number of rows in the original data
            x_names (list): list of names of the features (i.e. independent variables) to use
            y_name (string): name of the target (i.e. dependent variable) to use
            intercept (boolean): true if the lin-reg equation should have a y-intercept, false if not
        Returns:
            linear regression model (Dataframe) in the following format:
                Each independent variable gets its own row; there are two columns: 'Estimate' and 'Std. Error'.
                'Estimate' is the calculated coefficient for that row's corresponding independent variable,
                'Std. Error' is self evident.

                Here is an example return value given intercept=FALSE, independent variables 'Height' and 'Volume'
                and dependent variable 'Girth':

                           Estimate    Std. Error
                    Height -0.04548    0.02686
                    Volume  0.19518    0.01041
        """
    eigenvals, _ = list(np.linalg.eig(release.values))
    if not all([ev != 0 for ev in eigenvals]):
        # TODO throw error? Matrix is not invertible
        return None
    elif not all([ev > 0 for ev in eigenvals]):
        # TODO throw error? Matrix is not positive definite
        return None
    else:
        # Find locations corresponding to the given x & y names
        loc_vec = [False]*release.shape[0]
        row_labels = release.index.values
        x_loc = []
        y_loc = None
        for index in range(len(row_labels)):
            if row_labels[index] in x_names:
                loc_vec[index] = True
                x_loc.append(index)
            if row_labels[index] == y_name:
                y_loc = index
        if x_loc is None or y_loc is None:
            # TODO throw error or something? Names aren't found in the release
            return None

        # Use a sweep to find the coefficient of the independent variable in
        # the linear regression corresponding to the covariance matrix
        sweep = amsweep(release.values, num_rows, loc_vec) # TODO implement amsweep
        coef = sweep[y_loc, x_loc]

        # Calculate the standard error
        submatrix = release.values[x_loc, :][:, x_loc]
        se = list(map(np.sqrt, sweep[y_loc, y_loc] * np.diag(np.linalg.inv(submatrix))))

        new_x_names = [release.index.values[x_loc[i]] for i in range(len(x_loc))]

        def round_5(elem):
            return round(elem, 5)

        # Round both values to account for floating point error, put in Series
        estimates = pd.Series(map(round_5, coef), index=new_x_names, name='Estimate')
        std_error = pd.Series(map(round_5, se), index=new_x_names, name='Std. Error')

        return pd.DataFrame([estimates, std_error]).transpose()


def amsweep(release, num_rows, loc_vec):
    # TODO: impement sweep
    return None