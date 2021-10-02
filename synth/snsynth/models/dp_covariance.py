import pandas as pd
import numpy as np
import math


class DPcovariance:

    # Implementation is based off of https://github.com/privacytoolsproject/PSI-Library

    def __init__(self, n, cols, rng, global_eps, epsilon_dist=None, alpha=0.05):

        # TODO finish adding functionality for intercept
        intercept = False

        # The following variables are for different ways of setting up the epsilon value for DP covariance calculation
        # There is infrastructure for them, but we're currently choosing not to expose them.
        epsilon = None
        accuracy = None
        impute_rng = None
        accuracy_vals = None

        self.num_rows = n
        self.columns = cols

        self.intercept = intercept
        self.alpha = alpha

        self.rng = check_range(rng)
        self.sens = covariance_sensitivity(n, rng, intercept)

        if impute_rng is None:
            self.imputeRng = rng
        else:
            self.imputeRng = impute_rng

        if self.intercept:
            self.columns = ["intercept"] + self.columns
        else:
            self.columns = self.columns

        s = len(self.columns)
        output_length = (np.zeros((s, s))[np.tril_indices(s)]).size
        # Distribute epsilon across all covariances that will be calculated
        if epsilon is not None:
            self.epsilon = check_epsilon(epsilon, expected_length=output_length)
            self.globalEps = sum(self.epsilon)
        # Option 2: Enter global epsilon value and vector of percentages specifying how to split global
        # epsilon between covariance calculations.
        elif global_eps is not None and epsilon_dist is not None:
            self.globalEps = check_global_epsilon(global_eps)
            self.epsilonDist = check_epsilon_dist(epsilon_dist, output_length)
            self.epsilon = distribute_epsilon(self.globalEps, epsilon_dist=epsilon_dist)
            self.accuracyVals = laplace_get_accuracy(self.sens, self.epsilon, self.alpha)
        # Option 3: Only enter global epsilon, and have it be split evenly between covariance calculations.
        elif global_eps is not None:
            self.globalEps = check_global_epsilon(global_eps)
            self.epsilon = distribute_epsilon(self.globalEps, n_calcs=output_length)
            self.accuracyVals = laplace_get_accuracy(self.sens, self.epsilon, self.alpha)
        # Option 4: Enter an accuracy value instead of an epsilon, and calculate
        # individual epsilons with this accuracy.
        elif accuracy is not None:
            self.accuracy = check_accuracy(accuracy)
            self.epsilon = laplace_get_epsilon(self.sens, self.accuracy, self.alpha)
            self.globalEps = sum(self.epsilon)
        # Option 5: Enter vector of accuracy values, and calculate ith epsilon value from ith accuracy value
        elif accuracy_vals is not None:
            self.accuracyVals = check_accuracy_vals(accuracy_vals, output_length)
            self.epsilon = laplace_get_epsilon(self.sens, self.accuracyVals, self.alpha)
            self.globalEps = sum(self.epsilon)

    def make_covar_symmetric(self, covar):
        """
        Converts unique private covariances into symmetric matrix

        Args:
            covar (???): differentially privately release of elements in lower triangle of covariance matrix
        Returns:
            A symmetric differentially private covariance matrix (numpy array)
        """
        n = len(self.columns)
        indices = np.triu_indices(n)
        m = np.zeros((n, n))
        m[indices] = covar
        m = m.T
        m = np.tril(m) + np.triu(m.T, 1)
        df = pd.DataFrame(m, columns=self.columns, index=self.columns)
        return df

    def release(self, data):
        new_data = censor_data(data[self.columns], self.rng)
        new_data = fill_missing(new_data, impute_rng=self.imputeRng)

        # TODO: add intercept functionality
        def covar(x, intercept=False):
            if intercept:
                pass  # TODO: Find python equivalent for the following R code: `x < - cbind(1, x)`
            covariance = np.cov(x)
            return list(covariance[np.tril_indices(covariance.shape[0])])

        def q_lap_iter(p, mu=0, b=1):
            for i in range(len(p)):
                p[i] = q_lap(p[i], mu, b[i])
            return p

        def q_lap(elem, mu=0, b=1):
            if elem < 0.5:
                return mu + b * np.log(2 * elem)
            else:
                return mu - b * np.log(2 - 2 * elem)

        def dp_noise(n, noise_scale):
            u = np.random.uniform(size=n)
            return q_lap_iter(u, b=noise_scale)

        true_val = covar(data.values.T, self.intercept)
        scale = self.sens / self.epsilon
        val = np.array(true_val) + dp_noise(n=len(true_val), noise_scale=scale)
        return list(val)

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
            release (Dataframe): differentially privately released covariance matrix that will be used to
                make the linear regression
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
        loc_vec = [False] * release.shape[0]
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
        sweep = amsweep(release.values / num_rows, np.array(loc_vec))
        coef = sweep[y_loc, x_loc]

        # Calculate the standard error
        submatrix = release.values[x_loc, :][:, x_loc]
        se = list(map(np.sqrt, sweep[y_loc, y_loc] * np.diag(np.linalg.inv(submatrix))))

        new_x_names = [release.index.values[x_loc[i]] for i in range(len(x_loc))]

        def round_5(elem):
            return round(elem, 5)

        # Round both values to account for floating point error, put in Series
        estimates = pd.Series(map(round_5, coef), index=new_x_names, name="Estimate")
        std_error = pd.Series(map(round_5, se), index=new_x_names, name="Std. Error")

        return pd.DataFrame([estimates, std_error]).transpose()


def check_accuracy_vals(accuracy_vals, expected_length):
    if len(accuracy_vals) != expected_length:
        raise ValueError("Epsilon parameter has improper length")
    else:
        for eps in accuracy_vals:
            if eps <= 0:
                raise ValueError("Privacy parameter epsilon must be a value greater than zero")
    return accuracy_vals


def laplace_get_epsilon(sens, accuracy, alpha=0.05):
    return np.log(1 / alpha) * (sens / accuracy)


def check_accuracy(accuracy):
    if accuracy <= 0:
        raise ValueError("Privacy parameter epsilon must be a value greater than zero")
    return accuracy


def laplace_get_accuracy(sens, epsilon, alpha=0.05):
    return np.log(1 / alpha) * (sens / epsilon)


def distribute_epsilon(global_eps, n_calcs=None, epsilon_dist=None):
    if epsilon_dist is None:
        eps = [global_eps / n_calcs for i in range(n_calcs)]
    else:
        eps = [eps * global_eps for eps in epsilon_dist]
    return eps


def check_epsilon_dist(epsilon_dist, expected_length):
    if len(epsilon_dist) != expected_length:
        raise ValueError("Epsilon parameter has improper length")
    else:
        for eps in epsilon_dist:
            if eps <= 0:
                raise ValueError("All values in epsilonDist must be a value greater than zero")
        if sum(epsilon_dist) != 1.0:
            raise ValueError("All values in epsilonDist must sum to 1")
    return epsilon_dist


def check_epsilon(epsilon, expected_length):
    if len(epsilon) != expected_length:
        raise ValueError("Epsilon parameter has improper length")
    else:
        for eps in epsilon:
            if eps <= 0:
                raise ValueError("(Privacy parameter epsilon must be a value greater than zero")
            elif eps >= 3:
                raise ValueError("This is a higher global value than recommended for most cases")
    return epsilon


def check_global_epsilon(eps):
    if eps <= 0:
        raise ValueError("(Privacy parameter epsilon must be a value greater than zero")
    elif eps >= 3:
        raise ValueError("This is a higher global value than recommended for most cases")
    return eps


def covariance_sensitivity(n, rng, intercept):
    diffs = []
    for i in range(rng.shape[1]):
        diffs.append(rng[i][1] - rng[i][0])
    if intercept:
        diffs = [0] + diffs
    const = 2 / n
    sensitivity = []
    for i in range(len(diffs)):
        for j in range(i, len(diffs)):
            s = const * diffs[i] * diffs[j]
            sensitivity.append(s)
    return np.array(sensitivity)


def check_range(rng):
    rng.columns = list(range(rng.shape[1]))
    for col in range(rng.shape[1]):
        rng[col] = rng[col].sort_values()
    return rng


def fill_missing_1D(x, low, high):
    n_missing = x.isnull().sum()
    u = np.random.uniform(size=n_missing)

    def scale(v):
        return v * (high - low) + low

    u = list(map(scale, u))

    def replace_nan(v):
        if math.isnan(v):
            return u.pop()
        return v

    return x.apply(replace_nan)


def fill_missing(data, impute_rng):
    for i in range(data.shape[1]):
        data[i] = fill_missing_1D(data[i], impute_rng[i][0], impute_rng[i][1])
    return data


def censor(value, low, high):
    if value < low:
        return low
    elif value > high:
        return high
    else:
        return value


def censor_data_1D(x, ll, h):
    def scale(v):
        return censor(v, ll, h)

    return x.apply(scale)


def censor_data(data, rng):
    new_data = data

    new_data.columns = list(range(data.shape[1]))
    rng = check_range(rng)

    for i in range(data.shape[1]):
        data[i] = censor_data_1D(data[i], rng[i][0], rng[i][1])
    return data


def amsweep(g, m):
    """
    Sweeps a covariance matrix to extract regression coefficients.

    Args:
        g (Numpy array): a numeric, symmetric covariance matrix divided by the number of observations in the data
        m (Numpy array): a logical vector of length equal to the number of rows in g
        in which the True values correspond to the x values in the matrix
        and the False values correspond to the y values in the matrix

    Return:
        a matrix with the coefficients from g
    """
    # if m is a vector of all falses, then return g
    if np.array_equal(m, np.full(np.shape(m), False, dtype=bool)):
        return g
    else:
        p = np.shape(g)[
            0
        ]  # number of rows of g (np.shape gives a tuple as (rows, cols), so we index [0])
        rowsm = sum(m)  # sum of logical vector "m" (m must be a (n,) shape np array)

        # if all values of m are True (thus making the sum equal to the length),
        # we take the inverse and then negate all the values
        if p == rowsm:
            h = np.linalg.inv(g)  # inverse of g
            h = np.negative(h)  # negate the sign of all elements
        else:
            k = np.where(m)[0]  # indices where m is True
            kcompl = np.where(m is False)[0]  # indices where m is False

            # separate the elements of g
            # make the type np.matrix so that dimensions are preserved correctly
            g11 = np.matrix(g[k, k])
            g12 = np.matrix(g[k, kcompl])
            g21 = np.transpose(g12)
            g22 = np.matrix(g[kcompl, kcompl])

            # use a try-except to get the inverse of g11
            try:
                h11a = np.linalg.inv(g11)  # try to get the regular inverse
            except BaseException:  # should have LinAlgError (not defined error)
                h11a = np.linalg.pinv(g11)
            h11 = np.negative(h11a)

            # matrix multiplication to get sections of h
            h12 = np.matmul(h11a, g12)
            h21 = np.transpose(h12)
            h22 = g22 - np.matmul(np.matmul(g21, h11a), g12)

            # combine sections of h
            hwo = np.concatenate(
                (np.concatenate((h11, h12), axis=1), np.concatenate((h21, h22), axis=1)), axis=0
            )
            hwo = np.asarray(
                hwo
            )  # convert back to array (from matrix) to avoid weird indexing behavior
            xordering = np.concatenate((k, kcompl), axis=0)  # concatenate k and kcompl
            h = np.zeros((p, p))  # make a pxp array of zeros

            for i in range(p):  # traverse each element as defined by xordering
                for j in range(p):
                    h[xordering[i]][xordering[j]] = hwo[i][
                        j
                    ]  # and replace it with the normal i, j element from hwo

        return h
