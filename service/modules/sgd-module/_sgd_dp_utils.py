import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from sklearn.utils._seq_dataset import ArrayDataset32, CSRDataset32
from sklearn.utils._seq_dataset import ArrayDataset64, CSRDataset64
from sklearn.utils._seq_dataset import SequentialDataset64, SequentialDataset32
#from sklearn.utils._cython_blas import _scal, _axpy

SPARSE_INTERCEPT_DECAY = 0.01
# For sparse data intercept updates are scaled by this decay factor to avoid
# intercept oscillation.

INT_MAX = np.iinfo(np.int32).max

def _dot(x,y):
    return np.dot(np.transpose(x), y)


def _scal(alpha, x):
    return alpha * x


def _axpy(alpha, x, y):
    return alpha *x + y


class WeightVector:
    def __init__(self, w, aw):

        if w.shape[0] > INT_MAX:
            raise ValueError("More than %d features not supported; got %d."

                             % (INT_MAX, w.shape[0]))
        self.w = w
        self.aw = aw
        self.wscale = 1.0

        self.n_features = self.w.shape[0]

        self.sq_norm = _dot(self.w, self.w)




        self.w_data_ptr = self.w

        if aw is not None:
            self.aw_data_ptr = self.aw

            self.average_a = 0.0

            self.average_b = 1.0
        else:
            self.aw_data_ptr = None
            self.average_a = None

            self.average_b = None

    def reset_wscale(self):
        """Scales each coef of ``w`` by ``wscale`` and resets it to 1. """
        if self.aw_data_ptr is not None:
            self.aw_data_ptr = _axpy(self.average_a,
                                     self.w_data_ptr,
                                     self.aw_data_ptr)

            self.aw_data_ptr = _scal( 1.0 / self.average_b, self.aw_data_ptr)

            self.average_a = 0.0

            self.average_b = 1.0

        self.w_data_ptr = _scal(self.wscale, self.w_data_ptr)

        self.wscale = 1.0

    def add(self, x_data_ptr, x_ind_ptr, xnnz, c):

        """Scales sample x by constant c and adds it to the weight vector.
        This operation updates ``sq_norm``.

        Parameters

        ----------

        x_data_ptr : double*

            The array which holds the feature values of ``x``.

        x_ind_ptr : np.intc*

            The array which holds the feature indices of ``x``.

        xnnz : int

            The number of non-zero features of ``x``.

        c : double

            The scaling constant for the example.

        """

        innerprod = 0.0
        xsqnorm = 0.0

        # the next two lines save a factor of 2!
        wscale = self.wscale
        w_data_ptr = self.w_data_ptr

        for j in range(xnnz):
            idx = x_ind_ptr[j]

            val = x_data_ptr[j]

            innerprod += (w_data_ptr[idx] * val)

            xsqnorm += (val * val)

            w_data_ptr[idx] += val * (c / wscale)

        self.sq_norm += (xsqnorm * c * c) + (2.0 * innerprod * wscale * c)

        # Update the average weights according to the sparse trick defined

        # here: https://research.microsoft.com/pubs/192769/tricks-2012.pdf

        # by Leon Bottou

    def add_average(self, x_data_ptr, x_ind_ptr, xnnz, c, num_iter):

        """Updates the average weight vector.



        Parameters

        ----------

        x_data_ptr : double*

            The array which holds the feature values of ``x``.

        x_ind_ptr : np.intc*

            The array which holds the feature indices of ``x``.

        xnnz : int

            The number of non-zero features of ``x``.

        c : double

            The scaling constant for the example.

        num_iter : double

            The total number of iterations.

        """

        mu = 1.0 / num_iter
        average_a = self.average_a
        wscale = self.wscale
        aw_data_ptr = self.aw_data_ptr

        for j in range(xnnz):
            idx = x_ind_ptr[j]

            val = x_data_ptr[j]

            aw_data_ptr[idx] += (self.average_a * val * (-c / wscale))

        # Once the sample has been processed

        # update the average_a and average_b

        if num_iter > 1:
            self.average_b /= (1.0 - mu)

        self.average_a += mu * self.average_b * wscale

    def dot(self, x_data_ptr, x_ind_ptr, xnnz):

        """Computes the dot product of a sample x and the weight vector.

        Parameters

        ----------

        x_data_ptr : double*

            The array which holds the feature values of ``x``.

        x_ind_ptr : np.intc*

            The array which holds the feature indices of ``x``.

        xnnz : int

            The number of non-zero features of ``x`` (length of x_ind_ptr).



        Returns

        -------

        innerprod : double

            The inner product of ``x`` and ``w``.

        """

        innerprod = 0.0
        w_data_ptr = self.w_data_ptr

        for j in range(xnnz):

            idx = x_ind_ptr[j]
            innerprod += w_data_ptr[idx] * x_data_ptr[j]

        innerprod *= self.wscale

        return innerprod

    def scale(self, c):

        """Scales the weight vector by a constant ``c``.

        It updates ``wscale`` and ``sq_norm``. If ``wscale`` gets too

        small we call ``reset_swcale``."""

        self.wscale *= c

        self.sq_norm *= (c * c)

        if self.wscale < 1e-9:
            self.reset_wscale()

    def norm(self):

            """The L2 norm of the weight vector. """

            return np.sqrt(self.sq_norm)


class Dataset:
    def __init__(self, X, y, sample_weight, array_data, seed=1):
        self.X = X
        self.y = y
        self.sample_weights = sample_weight

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.dataset = array_data(X, y, sample_weight, seed=seed)

    def shuffle(self, seed):
        self.dataset._shuffle_py(seed)

    #params: x_data_ptr, x_ind_ptr, xnnz, y, sample_weight
    def next(self):
        return self.dataset._next_py()


def make_dataset_dp(X, y, sample_weight, random_state=None):
    """Create ``Dataset`` abstraction for sparse and dense inputs.

    This also returns the ``intercept_decay`` which is different
    for sparse datasets.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Training data

    y : array_like, shape (n_samples, )
        Target values.

    sample_weight : numpy array of shape (n_samples,)
        The weight of each sample

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    dataset
        The ``Dataset`` abstraction
    intercept_decay
        The intercept decay
    """

    rng = check_random_state(random_state)
    # seed should never be 0 in SequentialDataset64
    seed = rng.randint(1, np.iinfo(np.int32).max)

    if X.dtype == np.float32:
        CSRData = CSRDataset32
        ArrayData = ArrayDataset32
    else:
        CSRData = CSRDataset64
        ArrayData = ArrayDataset64

    if sp.issparse(X):
        dataset = CSRData(X.data, X.indptr, X.indices, y, sample_weight,
                          seed=seed)
        intercept_decay = SPARSE_INTERCEPT_DECAY
    else:
        X = np.ascontiguousarray(X)
        dataset = Dataset(X, y, sample_weight, ArrayData, seed=seed)
        intercept_decay = 1.0

    return dataset, intercept_decay
