# Authors: Peter Prettenhofer <peter.prettenhofer@gmail.com> (main author)
#          Mathieu Blondel (partial_fit support)
#
# License: BSD 3 clause
"""Classification and regression using Stochastic Gradient Descent (SGD)."""

import numpy as np
import warnings

from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
from _sgd_dp_utils import make_dataset_dp
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.linear_model._stochastic_gradient import BaseSGDClassifier, BaseSGDRegressor, _prepare_fit_binary
from sklearn.linear_model import SGDClassifier, SGDRegressor

from _sgd_dp_fast import plain_sgd, average_sgd
from sklearn.utils import compute_class_weight
from sklearn.linear_model._sgd_fast import Hinge
from sklearn.linear_model._sgd_fast import SquaredHinge
from sklearn.linear_model._sgd_fast import Log
from sklearn.linear_model._sgd_fast import ModifiedHuber
from sklearn.linear_model._sgd_fast import SquaredLoss
from sklearn.linear_model._sgd_fast import Huber
from sklearn.linear_model._sgd_fast import EpsilonInsensitive
from sklearn.linear_model._sgd_fast import SquaredEpsilonInsensitive
from sklearn.utils.fixes import _joblib_parallel_args

class MySquaredLoss(SquaredLoss):
    def loss(self, p, y):
        return 0.5 * (p - y) * (p - y)

    def _dloss(self, p, y):
        return p - y

    def __reduce__(self):
        return MySquaredLoss, ()

LEARNING_RATE_TYPES = {"constant": 1, "optimal": 2, "invscaling": 3,
                       "adaptive": 4, "pa1": 5, "pa2": 6}

PENALTY_TYPES = {"none": 0, "l2": 2, "l1": 1, "elasticnet": 3}

DEFAULT_EPSILON = 0.1
# Default value of ``epsilon`` parameter.

MAX_INT = np.iinfo(np.int32).max

# TODO actually implement DP versions of these functions
average_sgd_dp = average_sgd
plain_sgd_dp = plain_sgd


def fit_binary_dp(est, i, X, y, alpha, C, learning_rate, max_iter,
                  pos_weight, neg_weight, sample_weight, validation_mask=None,
                  random_state=None):
    """Fit a single binary classifier.

    The i'th class is considered the "positive" class.

    Parameters
    ----------
    est : Estimator object
        The estimator to fit

    i : int
        Index of the positive class

    X : numpy array or sparse matrix of shape [n_samples,n_features]
        Training data

    y : numpy array of shape [n_samples, ]
        Target values

    alpha : float
        The regularization parameter

    C : float
        Maximum step size for passive aggressive

    learning_rate : string
        The learning rate. Accepted values are 'constant', 'optimal',
        'invscaling', 'pa1' and 'pa2'.

    max_iter : int
        The maximum number of iterations (epochs)

    pos_weight : float
        The weight of the positive class

    neg_weight : float
        The weight of the negative class

    sample_weight : numpy array of shape [n_samples, ]
        The weight of each sample

    validation_mask : numpy array of shape [n_samples, ], default=None
        Precomputed validation mask in case _fit_binary is called in the
        context of a one-vs-rest reduction.

    random_state : int, RandomState instance, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    # if average is not true, average_coef, and average_intercept will be
    # unused
    y_i, coef, intercept, average_coef, average_intercept = \
        _prepare_fit_binary(est, y, i)
    assert y_i.shape[0] == y.shape[0] == sample_weight.shape[0]

    random_state = check_random_state(random_state)
    dataset, intercept_decay = make_dataset_dp(
        X, y_i, sample_weight, random_state=random_state)

    penalty_type = est._get_penalty_type(est.penalty)
    learning_rate_type = est._get_learning_rate_type(learning_rate)

    if validation_mask is None:
        validation_mask = est._make_validation_split(y_i)
    classes = np.array([-1, 1], dtype=y_i.dtype)
    validation_score_cb = est._make_validation_score_cb(
        validation_mask, X, y_i, sample_weight, classes=classes)

    # numpy mtrand expects a C long which is a signed 32 bit integer under
    # Windows
    seed = random_state.randint(MAX_INT)

    tol = est.tol if est.tol is not None else -np.inf

    if not est.average:
        result = plain_sgd_dp(coef, intercept, est.loss_function_,
                           penalty_type, alpha, C, est.l1_ratio,
                           dataset, validation_mask, est.early_stopping,
                           validation_score_cb, int(est.n_iter_no_change),
                           max_iter, tol, int(est.fit_intercept),
                           int(est.verbose), int(est.shuffle), seed,
                           pos_weight, neg_weight,
                           learning_rate_type, est.eta0,
                           est.power_t, est.t_, intercept_decay)

    else:
        standard_coef, standard_intercept, average_coef, average_intercept, \
        n_iter_ = average_sgd_dp(coef, intercept, average_coef,
                              average_intercept, est.loss_function_,
                              penalty_type, alpha, C, est.l1_ratio,
                              dataset, validation_mask, est.early_stopping,
                              validation_score_cb,
                              int(est.n_iter_no_change), max_iter, tol,
                              int(est.fit_intercept), int(est.verbose),
                              int(est.shuffle), seed, pos_weight,
                              neg_weight, learning_rate_type, est.eta0,
                              est.power_t, est.t_, intercept_decay,
                              est.average)

        if len(est.classes_) == 2:
            est.average_intercept_[0] = average_intercept
        else:
            est.average_intercept_[i] = average_intercept

        result = standard_coef, standard_intercept, n_iter_

    return result


class BaseDPSGDClassifier(BaseSGDClassifier):
    loss_functions = {
        "hinge": (Hinge, 1.0),
        "squared_hinge": (SquaredHinge, 1.0),
        "perceptron": (Hinge, 0.0),
        "log": (Log,),
        "modified_huber": (ModifiedHuber,),
        "squared_loss": (MySquaredLoss,),
        "huber": (Huber, DEFAULT_EPSILON),
        "epsilon_insensitive": (EpsilonInsensitive, DEFAULT_EPSILON),
        "squared_epsilon_insensitive": (SquaredEpsilonInsensitive,
                                        DEFAULT_EPSILON),
    }

    @abstractmethod
    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
                 shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, n_jobs=None,
                 random_state=None, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, early_stopping=False,
                 validation_fraction=0.1, n_iter_no_change=5,
                 class_weight=None, warm_start=False, average=False):

        super().__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, warm_start=warm_start,
            average=average)
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    # Overrides OG _partial_fit() so that it can call _fit_multiclass_dp() and _fit_binary_dp()
    def _partial_fit(self, X, y, alpha, C,
                     loss, learning_rate, max_iter,
                     classes, sample_weight,
                     coef_init, intercept_init):
        X, y = check_X_y(X, y, 'csr', dtype=np.float64, order="C",
                         accept_large_sparse=False)

        n_samples, n_features = X.shape

        _check_partial_fit_first_call(self, classes)

        n_classes = self.classes_.shape[0]

        # Allocate datastructures from input arguments
        self._expanded_class_weight = compute_class_weight(self.class_weight,
                                                           self.classes_, y)
        sample_weight = _check_sample_weight(sample_weight, X)

        if getattr(self, "coef_", None) is None or coef_init is not None:
            self._allocate_parameter_mem(n_classes, n_features,
                                         coef_init, intercept_init)
        elif n_features != self.coef_.shape[-1]:
            raise ValueError("Number of features %d does not match previous "
                             "data %d." % (n_features, self.coef_.shape[-1]))

        self.loss_function_ = self._get_loss_function(loss)
        if not hasattr(self, "t_"):
            self.t_ = 1.0

        # delegate to concrete training procedure
        if n_classes > 2:
            self._fit_multiclass_dp(X, y, alpha=alpha, C=C,
                                    learning_rate=learning_rate,
                                    sample_weight=sample_weight,
                                    max_iter=max_iter)
        elif n_classes == 2:
            self._fit_binary_dp(X, y, alpha=alpha, C=C,
                                learning_rate=learning_rate,
                                sample_weight=sample_weight,
                                max_iter=max_iter)
        else:
            raise ValueError(
                "The number of classes has to be greater than one;"
                " got %d class" % n_classes)

        return self

    def _fit_binary_dp(self, X, y, alpha, C, sample_weight,
                       learning_rate, max_iter):
        """Fit a binary classifier on X and y. """
        coef, intercept, n_iter_ = fit_binary_dp(self, 1, X, y, alpha, C,
                                                 learning_rate, max_iter,
                                                 self._expanded_class_weight[1],
                                                 self._expanded_class_weight[0],
                                                 sample_weight,
                                                 random_state=self.random_state)

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        # need to be 2d
        if self.average > 0:
            if self.average <= self.t_ - 1:
                self.coef_ = self.average_coef_.reshape(1, -1)
                self.intercept_ = self.average_intercept_
            else:
                self.coef_ = self.standard_coef_.reshape(1, -1)
                self.standard_intercept_ = np.atleast_1d(intercept)
                self.intercept_ = self.standard_intercept_
        else:
            self.coef_ = coef.reshape(1, -1)
            # intercept is a float, need to convert it to an array of length 1
            self.intercept_ = np.atleast_1d(intercept)

    def _fit_multiclass_dp(self, X, y, alpha, C, learning_rate,
                           sample_weight, max_iter):
        """Fit a multi-class classifier by combining binary classifiers

        Each binary classifier predicts one class versus all others. This
        strategy is called OvA (One versus All) or OvR (One versus Rest).
        """
        # Precompute the validation split using the multiclass labels
        # to ensure proper balancing of the classes.
        validation_mask = self._make_validation_split(y)

        # Use joblib to fit OvA in parallel.
        # Pick the random seed for each job outside of fit_binary to avoid
        # sharing the estimator random state between threads which could lead
        # to non-deterministic behavior
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(MAX_INT, size=len(self.classes_))
        result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                          **_joblib_parallel_args(require="sharedmem"))(
            delayed(fit_binary_dp)(self, i, X, y, alpha, C, learning_rate,
                                   max_iter, self._expanded_class_weight[i],
                                   1., sample_weight,
                                   validation_mask=validation_mask,
                                   random_state=seed)
            for i, seed in enumerate(seeds))

        # take the maximum of n_iter_ over every binary fit
        n_iter_ = 0.
        for i, (_, intercept, n_iter_i) in enumerate(result):
            self.intercept_[i] = intercept
            n_iter_ = max(n_iter_, n_iter_i)

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        if self.average > 0:
            if self.average <= self.t_ - 1.0:
                self.coef_ = self.average_coef_
                self.intercept_ = self.average_intercept_
            else:
                self.coef_ = self.standard_coef_
                self.standard_intercept_ = np.atleast_1d(self.intercept_)
                self.intercept_ = self.standard_intercept_


class DPSGDClassifier(BaseDPSGDClassifier, SGDClassifier):
    """Linear classifiers (SVM, logistic regression, a.o.) with SGD training.

    This estimator implements regularized linear models with stochastic
    gradient descent (SGD) learning: the gradient of the loss is estimated
    each sample at a time and the model is updated along the way with a
    decreasing strength schedule (aka learning rate). SGD allows minibatch
    (online/out-of-core) learning, see the partial_fit method.
    For best results using the default learning rate schedule, the data should
    have zero mean and unit variance.

    This implementation works with data represented as dense or sparse arrays
    of floating point values for the features. The model it fits can be
    controlled with the loss parameter; by default, it fits a linear support
    vector machine (SVM).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    Read more in the :ref:`User Guide <sgd>`.

    Parameters
    ----------
    loss : str, default='hinge'
        The loss function to be used. Defaults to 'hinge', which gives a
        linear SVM.

        The possible options are 'hinge', 'log', 'modified_huber',
        'squared_hinge', 'perceptron', or a regression loss: 'squared_loss',
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.

        The 'log' loss gives logistic regression, a probabilistic classifier.
        'modified_huber' is another smooth loss that brings tolerance to
        outliers as well as probability estimates.
        'squared_hinge' is like hinge but is quadratically penalized.
        'perceptron' is the linear loss used by the perceptron algorithm.
        The other losses are designed for regression but can be useful in
        classification as well; see SGDRegressor for a description.

    penalty : {'l2', 'l1', 'elasticnet'}, default='l2'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' might bring sparsity to the model (feature selection)
        not achievable with 'l2'.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term. Defaults to 0.0001.
        Also used to compute learning_rate when set to 'optimal'.

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
        Defaults to 0.15.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.

        .. versionadded:: 0.19

    tol : float, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
        epochs.

        .. versionadded:: 0.19

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    epsilon : float, default=0.1
        Epsilon in the epsilon-insensitive loss functions; only if `loss` is
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
        For 'huber', determines the threshold at which it becomes less
        important to get the prediction exactly right.
        For epsilon-insensitive, any differences between the current prediction
        and the correct label are ignored if they are less than this threshold.

    n_jobs : int, default=None
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        Used for shuffling the data, when ``shuffle`` is set to ``True``.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    learning_rate : str, default='optimal'
        The learning rate schedule:

        'constant':
            eta = eta0
        'optimal': [default]
            eta = 1.0 / (alpha * (t + t0))
            where t0 is chosen by a heuristic proposed by Leon Bottou.
        'invscaling':
            eta = eta0 / pow(t, power_t)
        'adaptive':
            eta = eta0, as long as the training keeps decreasing.
            Each time n_iter_no_change consecutive epochs fail to decrease the
            training loss by tol or fail to increase validation score by tol if
            early_stopping is True, the current learning rate is divided by 5.

    eta0 : double, default=0.0
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. The default value is 0.0 as eta0 is not used by
        the default schedule 'optimal'.

    power_t : double, default=0.5
        The exponent for inverse scaling learning rate [default 0.5].

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set aside
        a stratified fraction of training data as validation and terminate
        training when validation score is not improving by at least tol for
        n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    class_weight : dict, {class_label: weight} or "balanced", default=None
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes
        are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit`` will result in increasing the
        existing counter.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So ``average=10`` will begin averaging after seeing 10
        samples.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) if n_classes == 2 else \
            (n_classes, n_features)
        Weights assigned to the features.

    intercept_ : ndarray of shape (1,) if n_classes == 2 else (n_classes,)
        Constants in decision function.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
        For multiclass fits, it is the maximum over every binary fit.

    loss_function_ : concrete ``LossFunction``

    classes_ : array of shape (n_classes,)

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples)``.

    See Also
    --------
    sklearn.svm.LinearSVC: Linear support vector classification.
    LogisticRegression: Logistic regression.
    Perceptron: Inherits from SGDClassifier. ``Perceptron()`` is equivalent to
        ``SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant",
        penalty=None)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> Y = np.array([1, 1, 2, 2])
    >>> clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    >>> clf.fit(X, Y)
    SGDClassifier()

    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001, l1_ratio=0.15,
                 fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True,
                 verbose=0, epsilon=DEFAULT_EPSILON, n_jobs=None,
                 random_state=None, learning_rate="optimal", eta0=0.0,
                 power_t=0.5, early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, class_weight=None, warm_start=False,
                 average=False):
        super().__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon, n_jobs=n_jobs,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, class_weight=class_weight,
            warm_start=warm_start, average=average)


class BaseDPSGDRegressor(BaseSGDRegressor):
    loss_functions = {
        "squared_loss": (MySquaredLoss,),
        "huber": (Huber, DEFAULT_EPSILON),
        "epsilon_insensitive": (EpsilonInsensitive, DEFAULT_EPSILON),
        "squared_epsilon_insensitive": (SquaredEpsilonInsensitive,
                                        DEFAULT_EPSILON),
    }

    @abstractmethod
    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
                 shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON,
                 random_state=None, learning_rate="invscaling", eta0=0.01,
                 power_t=0.25, early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, warm_start=False, average=False):
        super().__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, warm_start=warm_start,
            average=average)

    def _partial_fit(self, X, y, alpha, C, loss, learning_rate,
                     max_iter, sample_weight, coef_init, intercept_init):
        X, y = check_X_y(X, y, "csr", copy=False, order='C', dtype=np.float64,
                         accept_large_sparse=False)
        y = y.astype(np.float64, copy=False)

        n_samples, n_features = X.shape

        sample_weight = _check_sample_weight(sample_weight, X)

        # Allocate datastructures from input arguments
        if getattr(self, "coef_", None) is None:
            self._allocate_parameter_mem(1, n_features, coef_init,
                                         intercept_init)
        elif n_features != self.coef_.shape[-1]:
            raise ValueError("Number of features %d does not match previous "
                             "data %d." % (n_features, self.coef_.shape[-1]))
        if self.average > 0 and getattr(self, "average_coef_", None) is None:
            self.average_coef_ = np.zeros(n_features,
                                          dtype=np.float64,
                                          order="C")
            self.average_intercept_ = np.zeros(1, dtype=np.float64, order="C")

        self._fit_regressor_dp(X, y, alpha, C, loss, learning_rate,
                               sample_weight, max_iter)

        return self

    def _fit_regressor_dp(self, X, y, alpha, C, loss, learning_rate,
                       sample_weight, max_iter):
        dataset, intercept_decay = make_dataset_dp(X, y, sample_weight)

        loss_function = self._get_loss_function(loss)
        penalty_type = self._get_penalty_type(self.penalty)
        learning_rate_type = self._get_learning_rate_type(learning_rate)

        if not hasattr(self, "t_"):
            self.t_ = 1.0

        validation_mask = self._make_validation_split(y)
        validation_score_cb = self._make_validation_score_cb(
            validation_mask, X, y, sample_weight)

        random_state = check_random_state(self.random_state)
        # numpy mtrand expects a C long which is a signed 32 bit integer under
        # Windows
        seed = random_state.randint(0, np.iinfo(np.int32).max)

        tol = self.tol if self.tol is not None else -np.inf

        if self.average > 0:
            self.standard_coef_, self.standard_intercept_, \
            self.average_coef_, self.average_intercept_, self.n_iter_ = \
                average_sgd_dp(self.standard_coef_,
                            self.standard_intercept_[0],
                            self.average_coef_,
                            self.average_intercept_[0],
                            loss_function,
                            penalty_type,
                            alpha, C,
                            self.l1_ratio,
                            dataset,
                            validation_mask, self.early_stopping,
                            validation_score_cb,
                            int(self.n_iter_no_change),
                            max_iter, tol,
                            int(self.fit_intercept),
                            int(self.verbose),
                            int(self.shuffle),
                            seed,
                            1.0, 1.0,
                            learning_rate_type,
                            self.eta0, self.power_t, self.t_,
                            intercept_decay, self.average)

            self.average_intercept_ = np.atleast_1d(self.average_intercept_)
            self.standard_intercept_ = np.atleast_1d(self.standard_intercept_)
            self.t_ += self.n_iter_ * X.shape[0]

            if self.average <= self.t_ - 1.0:
                self.coef_ = self.average_coef_
                self.intercept_ = self.average_intercept_
            else:
                self.coef_ = self.standard_coef_
                self.intercept_ = self.standard_intercept_

        else:
            self.coef_, self.intercept_, self.n_iter_ = \
                plain_sgd_dp(self.coef_,
                          self.intercept_[0],
                          loss_function,
                          penalty_type,
                          alpha, C,
                          self.l1_ratio,
                          dataset,
                          validation_mask, self.early_stopping,
                          validation_score_cb,
                          int(self.n_iter_no_change),
                          max_iter, tol,
                          int(self.fit_intercept),
                          int(self.verbose),
                          int(self.shuffle),
                          seed,
                          1.0, 1.0,
                          learning_rate_type,
                          self.eta0, self.power_t, self.t_,
                          intercept_decay)

            self.t_ += self.n_iter_ * X.shape[0]
            self.intercept_ = np.atleast_1d(self.intercept_)


class DPSGDRegressor(BaseDPSGDRegressor, SGDRegressor):
    """Linear model fitted by minimizing a regularized empirical loss with SGD

    SGD stands for Stochastic Gradient Descent: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a decreasing strength schedule (aka learning rate).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    This implementation works with data represented as dense numpy arrays of
    floating point values for the features.

    Read more in the :ref:`User Guide <sgd>`.

    Parameters
    ----------
    loss : str, default='squared_loss'
        The loss function to be used. The possible values are 'squared_loss',
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'

        The 'squared_loss' refers to the ordinary least squares fit.
        'huber' modifies 'squared_loss' to focus less on getting outliers
        correct by switching from squared to linear loss past a distance of
        epsilon. 'epsilon_insensitive' ignores errors less than epsilon and is
        linear past that; this is the loss function used in SVR.
        'squared_epsilon_insensitive' is the same but becomes squared loss past
        a tolerance of epsilon.

    penalty : {'l2', 'l1', 'elasticnet'}, default='l2'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' might bring sparsity to the model (feature selection)
        not achievable with 'l2'.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term.
        Also used to compute learning_rate when set to 'optimal'.

    l1_ratio : float, default=0.15
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, default=1000
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        :meth:`partial_fit` method.

        .. versionadded:: 0.19

    tol : float, default=1e-3
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
        epochs.

        .. versionadded:: 0.19

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : int, default=0
        The verbosity level.

    epsilon : float, default=0.1
        Epsilon in the epsilon-insensitive loss functions; only if `loss` is
        'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
        For 'huber', determines the threshold at which it becomes less
        important to get the prediction exactly right.
        For epsilon-insensitive, any differences between the current prediction
        and the correct label are ignored if they are less than this threshold.

    random_state : int, RandomState instance, default=None
        Used for shuffling the data, when ``shuffle`` is set to ``True``.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    learning_rate : string, default='invscaling'
        The learning rate schedule:

        'constant':
            eta = eta0
        'optimal':
            eta = 1.0 / (alpha * (t + t0))
            where t0 is chosen by a heuristic proposed by Leon Bottou.
        'invscaling': [default]
            eta = eta0 / pow(t, power_t)
        'adaptive':
            eta = eta0, as long as the training keeps decreasing.
            Each time n_iter_no_change consecutive epochs fail to decrease the
            training loss by tol or fail to increase validation score by tol if
            early_stopping is True, the current learning rate is divided by 5.

    eta0 : double, default=0.01
        The initial learning rate for the 'constant', 'invscaling' or
        'adaptive' schedules. The default value is 0.01.

    power_t : double, default=0.25
        The exponent for inverse scaling learning rate.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate
        training when validation score is not improving by at least tol for
        n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.
        If a dynamic learning rate is used, the learning rate is adapted
        depending on the number of samples already seen. Calling ``fit`` resets
        this counter, while ``partial_fit``  will result in increasing the
        existing counter.

    average : bool or int, default=False
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So ``average=10`` will begin averaging after seeing 10
        samples.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weights assigned to the features.

    intercept_ : ndarray of shape (1,)
        The intercept term.

    average_coef_ : ndarray of shape (n_features,)
        Averaged weights assigned to the features.

    average_intercept_ : ndarray of shape (1,)
        The averaged intercept term.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    t_ : int
        Number of weight updates performed during training.
        Same as ``(n_iter_ * n_samples)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
    >>> clf.fit(X, y)
    SGDRegressor()

    See also
    --------
    Ridge, ElasticNet, Lasso, sklearn.svm.SVR

    """

    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=1e-3,
                 shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON,
                 random_state=None, learning_rate="invscaling", eta0=0.01,
                 power_t=0.25, early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, warm_start=False, average=False):
        super().__init__(
            loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
            fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
            shuffle=shuffle, verbose=verbose, epsilon=epsilon,
            random_state=random_state, learning_rate=learning_rate, eta0=eta0,
            power_t=power_t, early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, warm_start=warm_start,
            average=average)


if __name__ == '__main__':
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    y = rng.randn(n_samples)
    X = rng.randn(n_samples, n_features)
    clf = SGDRegressor(max_iter=1000, tol=1e-3)
    clf.fit(X, y)
    print("SGDRegressor score:", clf.score(X, y))

    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    y = rng.randn(n_samples)
    X = rng.randn(n_samples, n_features)
    clf = DPSGDRegressor(max_iter=1000, tol=1e-3)
    clf.fit(X, y)
    print("DPSGDRegressor score:", clf.score(X, y))

    # X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    # Y = np.array([1, 1, 2, 2])
    # clf = SGDClassifier(max_iter=1000, tol=1e-3)
    # def break_things(*args, **kwargs):
    #     raise Exception("It worked")
    # import sklearn.linear_model
    # sklearn.linear_model._sgd_fast.plain_sgd = break_things
    # clf.fit(X, Y)
    # print("DPSGDClassifier score:", clf.score(X, Y))

