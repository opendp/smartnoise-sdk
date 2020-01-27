import sklearn.linear_model
from sklearn.linear_model._sgd_fast import *


def plain_sgd(weights,
              intercept,
              loss,
              penalty_type,
              alpha, C,
              l1_ratio,
              dataset,
              validation_mask,
              early_stopping, validation_score_cb,
              n_iter_no_change,
              max_iter, tol, fit_intercept,
              verbose, shuffle, seed,
              weight_pos, weight_neg,
              learning_rate, eta0,
              power_t,
              t=1.0,
              intercept_decay=1.0):
    """Plain SGD for generic loss functions and penalties.
    Parameters
    ----------
    weights : ndarray[double, ndim=1]
        The allocated coef_ vector.
    intercept : double
        The initial intercept.
    loss : LossFunction
        A concrete ``LossFunction`` object.
    penalty_type : int
        The penalty 2 for L2, 1 for L1, and 3 for Elastic-Net.
    alpha : float
        The regularization parameter.
    C : float
        Maximum step size for passive aggressive.
    l1_ratio : float
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
    dataset : SequentialDataset
        A concrete ``SequentialDataset`` object.
    validation_mask : ndarray[unsigned char, ndim=1]
        Equal to True on the validation set.
    early_stopping : boolean
        Whether to use a stopping criterion based on the validation set.
    validation_score_cb : callable
        A callable to compute a validation score given the current
        coefficients and intercept values.
        Used only if early_stopping is True.
    n_iter_no_change : int
        Number of iteration with no improvement to wait before stopping.
    max_iter : int
        The maximum number of iterations (epochs).
    tol: double
        The tolerance for the stopping criterion.
    fit_intercept : int
        Whether or not to fit the intercept (1 or 0).
    verbose : int
        Print verbose output; 0 for quite.
    shuffle : boolean
        Whether to shuffle the training data before each epoch.
    weight_pos : float
        The weight of the positive class.
    weight_neg : float
        The weight of the negative class.
    seed : np.uint32_t
        Seed of the pseudorandom number generator used to shuffle the data.
    learning_rate : int
        The learning rate:
        (1) constant, eta = eta0
        (2) optimal, eta = 1.0/(alpha * t).
        (3) inverse scaling, eta = eta0 / pow(t, power_t)
        (4) adaptive decrease
        (5) Passive Aggressive-I, eta = min(alpha, loss/norm(x))
        (6) Passive Aggressive-II, eta = 1.0 / (norm(x) + 0.5*alpha)
    eta0 : double
        The initial learning rate.
    power_t : double
        The exponent for inverse scaling learning rate.
    t : double
        Initial state of the learning rate. This value is equal to the
        iteration count except when the learning rate is set to `optimal`.
        Default: 1.0.
    intercept_decay : double
        The decay ratio of intercept, used in updating intercept.
    Returns
    -------
    weights : array, shape=[n_features]
        The fitted weight vector.
    intercept : float
        The fitted intercept term.
    n_iter_ : int
        The actual number of iter (epochs).
    """
    standard_weights, standard_intercept,\
        _, _, n_iter_ = _plain_sgd(weights,
                                   intercept,
                                   None,
                                   0,
                                   loss,
                                   penalty_type,
                                   alpha, C,
                                   l1_ratio,
                                   dataset,
                                   validation_mask,
                                   early_stopping,
                                   validation_score_cb,
                                   n_iter_no_change,
                                   max_iter, tol, fit_intercept,
                                   verbose, shuffle, seed,
                                   weight_pos, weight_neg,
                                   learning_rate, eta0,
                                   power_t,
                                   t,
                                   intercept_decay,
                                   0)
    return standard_weights, standard_intercept, n_iter_


def average_sgd(weights,
                intercept,
                average_weights,
                average_intercept,
                loss,
                penalty_type,
                alpha, C,
                l1_ratio,
                dataset,
                validation_mask,
                early_stopping, validation_score_cb,
                n_iter_no_change,
                max_iter, tol, fit_intercept,
                verbose, shuffle, seed,
                weight_pos, weight_neg,
                learning_rate, eta0,
                power_t,
                t=1.0,
                intercept_decay=1.0,
                average=1):
    """Average SGD for generic loss functions and penalties.
    Parameters
    ----------
    weights : ndarray[double, ndim=1]
        The allocated coef_ vector.
    intercept : double
        The initial intercept.
    average_weights : ndarray[double, ndim=1]
        The average weights as computed for ASGD
    average_intercept : double
        The average intercept for ASGD
    loss : LossFunction
        A concrete ``LossFunction`` object.
    penalty_type : int
        The penalty 2 for L2, 1 for L1, and 3 for Elastic-Net.
    alpha : float
        The regularization parameter.
    C : float
        Maximum step size for passive aggressive.
    l1_ratio : float
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
    dataset : SequentialDataset
        A concrete ``SequentialDataset`` object.
    validation_mask : ndarray[unsigned char, ndim=1]
        Equal to True on the validation set.
    early_stopping : boolean
        Whether to use a stopping criterion based on the validation set.
    validation_score_cb : callable
        A callable to compute a validation score given the current
        coefficients and intercept values.
        Used only if early_stopping is True.
    n_iter_no_change : int
        Number of iteration with no improvement to wait before stopping.
    max_iter : int
        The maximum number of iterations (epochs).
    tol: double
        The tolerance for the stopping criterion.
    dataset : SequentialDataset
        A concrete ``SequentialDataset`` object.
    fit_intercept : int
        Whether or not to fit the intercept (1 or 0).
    verbose : int
        Print verbose output; 0 for quite.
    shuffle : boolean
        Whether to shuffle the training data before each epoch.
    weight_pos : float
        The weight of the positive class.
    weight_neg : float
        The weight of the negative class.
    seed : np.uint32_t
        Seed of the pseudorandom number generator used to shuffle the data.
    learning_rate : int
        The learning rate:
        (1) constant, eta = eta0
        (2) optimal, eta = 1.0/(alpha * t).
        (3) inverse scaling, eta = eta0 / pow(t, power_t)
        (4) adaptive decrease
        (5) Passive Aggressive-I, eta = min(alpha, loss/norm(x))
        (6) Passive Aggressive-II, eta = 1.0 / (norm(x) + 0.5*alpha)
    eta0 : double
        The initial learning rate.
    power_t : double
        The exponent for inverse scaling learning rate.
    t : double
        Initial state of the learning rate. This value is equal to the
        iteration count except when the learning rate is set to `optimal`.
        Default: 1.0.
    average : int
        The number of iterations before averaging starts. average=1 is
        equivalent to averaging for all iterations.
    Returns
    -------
    weights : array, shape=[n_features]
        The fitted weight vector.
    intercept : float
        The fitted intercept term.
    average_weights : array shape=[n_features]
        The averaged weights across iterations
    average_intercept : float
        The averaged intercept across iterations
    n_iter_ : int
        The actual number of iter (epochs).
    """
    return _plain_sgd(weights,
                      intercept,
                      average_weights,
                      average_intercept,
                      loss,
                      penalty_type,
                      alpha, C,
                      l1_ratio,
                      dataset,
                      validation_mask,
                      early_stopping,
                      validation_score_cb,
                      n_iter_no_change,
                      max_iter, tol, fit_intercept,
                      verbose, shuffle, seed,
                      weight_pos, weight_neg,
                      learning_rate, eta0,
                      power_t,
                      t,
                      intercept_decay,
                      average)


def _plain_sgd(weights,
               intercept,
               average_weights,
               average_intercept,
               loss,
               penalty_type,
               alpha,
               C,
               l1_ratio,
               dataset,
               validation_mask,
               early_stopping,
               validation_score_cb,
               n_iter_no_change,
               max_iter,
               tol,
               fit_intercept,
               verbose,
               shuffle,
               seed,
               weight_pos,
               weight_neg,
               learning_rate,
               eta0,
               power_t,
               t=1.0,
               intercept_decay=1.0,
               average=0):

    # get the data information into easy vars
    n_samples = dataset.n_samples
    n_features = weights.shape[0]

    w = WeightVector(weights, average_weights)
    w_ptr = weights[0]
    x_data_ptr = None
    x_ind_ptr = None
    ps_ptr = None

    # helper variables
    no_improvement_count = 0
    infinity = False
    xnnz = None
    eta = 0.0
    p = 0.0
    update = 0.0
    sumloss = 0.0
    score = 0.0
    best_loss = float("inf")
    best_score = float("-inf")
    y = 0.0
    sample_weight
    class_weight = 1.0
    count = 0
    epoch = 0
    i = 0
    is_hinge = False
    optimal_init = 0.0
    dloss = 0.0
    MAX_DLOSS = 1e12
    max_change = 0.0
    max_weight = 0.0

    sample_index = int()
    validation_mask_view = validation_mask

    # q vector is only used for L1 regularization
    q = None
    q_data_ptr = None
    if penalty_type == L1 or penalty_type == ELASTICNET:
        q = np.zeros((n_features,), dtype=np.float64, order="c")
        q_data_ptr = q.data
    u = 0.0

    if penalty_type == L2:
        l1_ratio = 0.0
    elif penalty_type == L1:
        l1_ratio = 1.0

    eta = eta0

    if learning_rate == OPTIMAL:
        typw = np.sqrt(1.0 / np.sqrt(alpha))
        # computing eta0, the initial learning rate
        initial_eta0 = typw / max(1.0, loss.dloss(-typw, 1.0))
        # initialize t such that eta at first sample equals eta0
        optimal_init = 1.0 / (initial_eta0 * alpha)

    t_start = time()
    with nogil:
        for epoch in range(max_iter):
            sumloss = 0
            if verbose > 0:
                with gil:
                    print("-- Epoch %d" % (epoch + 1))
            if shuffle:
                dataset.shuffle(seed)
            for i in range(n_samples):
                dataset.next(x_data_ptr, x_ind_ptr, xnnz,
                             y, sample_weight)

                sample_index = dataset.index_data_ptr[dataset.current_index]
                if validation_mask_view[sample_index]:
                    # do not learn on the validation set
                    continue

                p = w.dot(x_data_ptr, x_ind_ptr, xnnz) + intercept
                if learning_rate == OPTIMAL:
                    eta = 1.0 / (alpha * (optimal_init + t - 1))
                elif learning_rate == INVSCALING:
                    eta = eta0 / pow(t, power_t)

                if verbose or not early_stopping:
                    sumloss += loss.loss(p, y)

                if y > 0.0:
                    class_weight = weight_pos
                else:
                    class_weight = weight_neg

                if learning_rate == PA1:
                    update = sqnorm(x_data_ptr, x_ind_ptr, xnnz)
                    if update == 0:
                        continue
                    update = min(C, loss.loss(p, y) / update)
                elif learning_rate == PA2:
                    update = sqnorm(x_data_ptr, x_ind_ptr, xnnz)
                    update = loss.loss(p, y) / (update + 0.5 / C)
                else:
                    dloss = loss._dloss(p, y)
                    # clip dloss with large values to avoid numerical
                    # instabilities
                    if dloss < -MAX_DLOSS:
                        dloss = -MAX_DLOSS
                    elif dloss > MAX_DLOSS:
                        dloss = MAX_DLOSS
                    update = -eta * dloss

                if learning_rate >= PA1:
                    if is_hinge:
                        # classification
                        update *= y
                    elif y - p < 0:
                        # regression
                        update *= -1

                update *= class_weight * sample_weight

                if penalty_type >= L2:
                    # do not scale to negative values when eta or alpha are too
                    # big: instead set the weights to zero
                    w.scale(max(0, 1.0 - ((1.0 - l1_ratio) * eta * alpha)))
                if update != 0.0:
                    w.add(x_data_ptr, x_ind_ptr, xnnz, update)
                    if fit_intercept == 1:
                        intercept += update * intercept_decay

                if 0 < average <= t:
                    # compute the average for the intercept and update the
                    # average weights, this is done regardless as to whether
                    # the update is 0

                    w.add_average(x_data_ptr, x_ind_ptr, xnnz,
                                  update, (t - average + 1))
                    average_intercept += ((intercept - average_intercept) /
                                          (t - average + 1))

                if penalty_type == L1 or penalty_type == ELASTICNET:
                    u += (l1_ratio * eta * alpha)
                    l1penalty(w, q_data_ptr, x_ind_ptr, xnnz, u)

                t += 1
                count += 1

            # report epoch information
            if verbose > 0:
                with gil:
                    print("Norm: %.2f, NNZs: %d, Bias: %.6f, T: %d, "
                          "Avg. loss: %f"
                          % (w.norm(), weights.nonzero()[0].shape[0],
                             intercept, count, sumloss / n_samples))
                    print("Total training time: %.2f seconds."
                          % (time() - t_start))

            # floating-point under-/overflow check.
            if (not np.isfinite(intercept)
                or any_nonfinite(weights.data, n_features)):
                infinity = True
                break

            # evaluate the score on the validation set
            if early_stopping:
                with gil:
                    score = validation_score_cb(weights, intercept)
                if tol > float("-inf") and score < best_score + tol:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                if score > best_score:
                    best_score = score
            # or evaluate the loss on the training set
            else:
                if tol > float("-inf") and sumloss > best_loss - tol * n_samples:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                if sumloss < best_loss:
                    best_loss = sumloss

            # if there is no improvement several times in a row
            if no_improvement_count >= n_iter_no_change:
                if learning_rate == ADAPTIVE and eta > 1e-6:
                    eta = eta / 5
                    no_improvement_count = 0
                else:
                    if verbose:
                        with gil:
                            print("Convergence after %d epochs took %.2f "
                                  "seconds" % (epoch + 1, time() - t_start))
                    break

    if infinity:
        raise ValueError(("Floating-point under-/overflow occurred at epoch"
                          " #%d. Scaling input data with StandardScaler or"
                          " MinMaxScaler might help.") % (epoch + 1))

    w.reset_wscale()

    return weights, intercept, average_weights, average_intercept, epoch + 1
import pdb;pdb.set_trace()
