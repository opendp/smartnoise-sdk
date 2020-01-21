import pandas as pd
import numpy as np
import math

class DPcovariance():

    # TODO: check out https://github.com/privacytoolsproject/PSI-Library/blob/2b71facd614845e548be9459ca94bc979eed0d4f/R/statistic-covariance.R#L101
    #   & add properties & etc

    def __init__(self, n, cols, rng, intercept=False, epsilon=None, globalEps=None, epsilonDist= None,
                 accuracy=None, accuracyVals=None, imputeRng=None,
                 alpha=0.05):
        self.num_rows = n
        self.columns = cols

        self.intercept = intercept
        self.alpha = alpha

        self.rng = check_range(rng)
        self.sens = covariance_sensitivity(n, rng, intercept)

        if imputeRng is None:
            self.imputeRng = rng
        else:
            self.imputeRng = imputeRng

        if self.intercept:
            self.columns = ['intercept']+self.columns
        else:
            self.columns = self.columns

        output_length = np.triu_indices(len(self.columns))
        # Distribute epsilon across all covariances that will be calculated
        if epsilon is not None:
            self.epsilon = check_epsilon(epsilon, expected_length=output_length)
            self.globalEps = sum(self.epsilon)
        # Option 2: Enter global epsilon value and vector of percentages specifying how to split global
        # epsilon between covariance calculations.
        elif globalEps is not None and epsilonDist is not None:
            self.globalEps = check_epsilon(globalEps)
            self.epsilonDist = check_epsilon_dist(epsilonDist, output_length)
            self.epsilon = distribute_epsilon(self.globalEps, epsilonDist=epsilonDist)
            self.accuracyVals = laplace_get_accuracy(self.sens, self.epsilon, self.alpha)
        # Option 3: Only enter global epsilon, and have it be split evenly between covariance calculations.
        elif globalEps is not None:
            self.globalEps = check_epsilon(globalEps)
            self.epsilon = distribute_epsilon(self.globalEps, nCalcs=output_length)
            self.accuracyVals = laplace_get_accuracy(self.sens, self.epsilon, self.alpha)
        # Option 4: Enter an accuracy value instead of an epsilon, and calculate individual epsilons with this accuracy.
        elif accuracy is not None:
            self.accuracy = check_accuracy(accuracy)
            self.epsilon = laplace_get_epsilon(self.sens, self.accuracy, self.alpha)
            self.globalEps = sum(self.epsilon)
        # Option 5: Enter vector of accuracy values, and calculate ith epsilon value from ith accuracy value
        elif accuracyVals is not None:
            self.accuracyVals = check_accuracy_vals(accuracyVals, output_length)
            self.epsilon = laplace_get_epsilon(self.sens, self.accuracyVals, self.alpha)
            self.globalEps = sum(self.epsilon)

    def make_covar_symmetric(self, covar):
        """
        Converts unique private covariances into symmetric matrix

        Args:
            covar (???): differentially privately release of elements in lower triangle of covariance matrix
            columns (list): a list of columns (x & y) to be included in the output
        Returns:
            A symmetric differentially private covariance matrix (Dataframe)
        """
        n = len(self.columns)
        indices = np.triu_indices(n)
        m = np.zeros((n, n))
        for i in range(indices):
            m[indices[i]] = covar[i]
        return np.tril(m) + np.triu(m.T, 1)

    def release(self, data):
        new_data = censor_data(data[self.columns], self.rng)
        new_data = fill_missing(new_data, imputeRng=self.imputeRng)

        # TODO: add intercept functionality
        def covar(x, intercept=False):
            if intercept:
                pass  # TODO: Find python equivalent for the following R code: `x < - cbind(1, x)`
            covariance = np.cov(x)
            return list(covariance[np.triu_indices(covariance.shape[0])])

        def q_lap(p, mu=0, b=1):
            if p < 0.5:
                return mu + b * np.log(2*p)
            else:
                return mu - b * np.log(2 - 2*p)

        def dp_noise(n, noise_scale):
            u = np.random.uniform(size=n)
            return q_lap(u, b=noise_scale)

        def sum_lists(first, second):
            return [x + y for x, y in zip(first, second)]

        true_val = covar(data)
        scale = self.sens / self.epsilon
        return sum_lists(true_val + dp_noise(n=len(true_val), noise_scale=scale))

    # TODO: this implementation only works for one dependent variable right now
    def get_linear_regression(self, data, x_names, y_name, intercept=False):
        """
        Takes in data, lists of feature names and target names, and whether or not
        we should calculate a y-intercept; and returns a DP linear regression model

        Args:
            data (Dataframe): the data that will be used to make the linear regression
            x_names (list): list of names of the features (i.e. independent variables) to use
            y_name (string): name of the target (i.e. dependent variable) to use
            intercept (boolean): true if the lin-reg equation should have a y-intercept, false if not
        Return:
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
        covar_matrix = self.make_covar_symmetric(self.release(data))
        return cov_method_lin_reg(covar_matrix, self.num_rows, x_names, y_name, intercept)


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
        raise ValueError("Matrix is not invertible")
    elif not all([ev > 0 for ev in eigenvals]):
        raise ValueError("Matrix is not positive definite")
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
            raise ValueError("Names aren't found in the release")

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


def check_accuracy_vals(accuracyVals, expected_length):
    if len(accuracyVals) != expected_length:
        raise ValueError("Epsilon parameter has improper length")
    else:
        for eps in accuracyVals:
            if eps <= 0:
                raise ValueError("Privacy parameter epsilon must be a value greater than zero")
    return accuracyVals


def laplace_get_epsilon(sens, accuracy, alpha=.05):
    return np.log(1/alpha) * (sens / accuracy)


def check_accuracy(accuracy):
    if accuracy <= 0:
        raise ValueError("Privacy parameter epsilon must be a value greater than zero")
    return accuracy


def laplace_get_accuracy(sens, epsilon, alpha=.05):
    return np.log(1/alpha) * (sens / epsilon)


def distribute_epsilon(globalEps, nCalcs=None, epsilonDist=None):
    if epsilonDist is None:
        eps = [globalEps/nCalcs for i in range(nCalcs)]
    else:
        eps = [eps*globalEps for eps in epsilonDist]
    return eps


def check_epsilon_dist(epsilonDist, expected_length):
    if len(epsilonDist) != expected_length:
        raise ValueError("Epsilon parameter has improper length")
    else:
        for eps in epsilonDist:
            if eps <= 0:
                raise ValueError("All values in epsilonDist must be a value greater than zero")
        if sum(epsilonDist) != 1.0:
            raise ValueError("All values in epsilonDist must sum to 1")
    return epsilonDist


def check_epsilon(epsilon, expected_length):
    if len(epsilon) != expected_length:
        raise ValueError("Epsilon parameter has improper length")
    else:
        for eps in epsilon:
            if eps <= 0:
                raise ValueError("(Privacy parameter epsilon must be a value greater than zero")"
            elif eps >= 3:
                raise ValueError("This is a higher global value than recommended for most cases")"
    return epsilon


def covariance_sensitivity(n, rng, intercept):
    diffs = []
    for i in range(rng.shape[1]):
        diffs.append(rng[i][0]-rng[i][1])
    if intercept:
        diffs = [0] + diffs
    const = 2/n
    sensitivity = []
    for i in range(len(diffs)):
        for j in range(len(diffs)):
            s = const * diffs[i] * diffs[j]
            sensitivity.append(s)
    return np.array(sensitivity)


def check_range(rng):
    for col in range(rng.shape[1]):
        rng[col] = rng[col].sort_values()
    return rng


def fill_missing_1D(x, l, h):
    n_missing = x.isnull().sum()
    u = np.random.uniform(size=n_missing)

    def scale(v):
        return v * (h - l) + l

    u = list(map(scale, u))

    def replace_nan(v):
        if math.isnan(v):
            return u.pop()
        return v

    return x.apply(replace_nan)


def fill_missing(data, imputeRng):
    for i in range(data.shape[1]):
        data[i] = fill_missing_1D(data[i], imputeRng[i][0], imputeRng[i][1])
    return data


def censor(value, low, high):
    if value < low:
        return low
    elif value > high:
        return high
    else:
        return value


def censor_data_1D(x, l, h):
    def scale(v):
        return censor(v, l, h)

    return x.apply(scale)


def censor_data(data, rng):
    check_range(rng)
    for i in range(data.shape[1]):
        data[i] = censor_data_1D(data[i], rng[i][0], rng[i][1])
    return data


def amsweep(release, num_rows, loc_vec):
    # TODO: impement sweep
    return None


# if __name__ == "__main__":
#     df = pd.DataFrame([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8], [7, 8, 9, 10]])
#     df.iloc[1, 1] = math.nan
#     df.iloc[0, 2] = math.nan
#     df.iloc[2, 3] = math.nan
#     data = df[[0,1,2,3]]
#     print("data:\n", data)
#     rng = pd.DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]]).transpose()
#     print("rng:\n", rng)
#     new_data = censor_data(data, rng)
#     new_data = fill_missing(new_data, imputeRng=rng)
#     print(new_data)
