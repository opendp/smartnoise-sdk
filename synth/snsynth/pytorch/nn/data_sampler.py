import warnings
import numpy as np
from opendp.mod import binary_search_param, enable_features
from opendp.trans import make_bounded_sum, make_clamp
from opendp.meas import make_base_geometric


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN.

    :param data(np.ndarray): The data to be conditionally sampled.
    :param output_info(list): The output information of the model.
    :param log_frequency(bool): Whether to use log frequency.
    :param per_column_epsilon(float): The privacy budget for each column.
    :param discrete_column_category_prob(list): The category probabilities for each discrete column.
        Use this to pass in cached noisy probabilities.  Data must match the schema of the original data.
    """

    def __init__(
            self,
            data,
            output_info,
            log_frequency,
            *ignore,
            per_column_epsilon=None,
            discrete_column_category_prob=None,
            **kwargs):
        self._data = data
        self._per_column_epsilon = per_column_epsilon

        if per_column_epsilon:
            if per_column_epsilon <= 0.0:
                raise ValueError("per_column_epsilon must be positive")
            bounds = (0, 1)
            max_contrib = 1
            enable_features('contrib')
            bounded_sum = (
                make_clamp(bounds=bounds) >>
                make_bounded_sum(bounds=bounds)
            )
            discovered_scale = binary_search_param(
                lambda s: bounded_sum >> make_base_geometric(scale=s),
                d_in=max_contrib,
                d_out=(self._per_column_epsilon))
            self._per_column_scale = discovered_scale
        else:
            self._per_column_scale = None
            if discrete_column_category_prob is None:
                warnings.warn(
                    "per_column_epsilon is not set, and no cached probabilites have been provided. "
                    "Sampler will not privatize frequencies, which may cause privacy leaks"
                )

        def is_discrete_column(column_info):
            return (len(column_info) == 1
                    and column_info[0].activation_fn == "softmax")

        n_discrete_columns = sum(
            [1 for column_info in output_info if is_discrete_column(column_info)])

        self._discrete_column_matrix_st = np.zeros(
            n_discrete_columns, dtype="int32")

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max(
            [column_info[0].dim for column_info in output_info
             if is_discrete_column(column_info)], default=0)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(
            n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros(
            (n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum(
            [column_info[0].dim for column_info in output_info
             if is_discrete_column(column_info)])

        eps_tot = 0.0
        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                # insert privacy here
                if self._per_column_scale:
                    geom = make_base_geometric(self._per_column_scale)
                    category_freq = [geom(int(v)) for v in category_freq]
                    eps_tot += self._per_column_epsilon
                category_freq = [1 if v < 1 else v for v in category_freq]
                if np.sum(category_freq) < 100:
                    # not enough data; use uniform distribution
                    category_freq = [1 for _ in category_freq]
                category_freq = np.array(category_freq, dtype='float64')
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, :span_info.dim] = (
                    category_prob)
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        self.total_spent = eps_tot

        if discrete_column_category_prob is not None:
            assert len(discrete_column_category_prob) == n_discrete_columns
            for i in range(n_discrete_columns):
                self._discrete_column_category_prob[i, :] = discrete_column_category_prob[i]
            self.total_spent = 0.0  # don't have to pay for cached noise

    @property
    def discrete_column_category_prob(self):
        return self._discrete_column_category_prob

    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(
            np.arange(self._n_discrete_columns), batch)

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = (self._discrete_column_cond_st[discrete_column_id]
                       + category_id_in_col)
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')

        for i in range(batch):
            row_idx = np.random.randint(0, len(self._data))
            col_idx = np.random.randint(0, self._n_discrete_columns)
            matrix_st = self._discrete_column_matrix_st[col_idx]
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

        return cond

    def sample_data(self, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            if len(self._rid_by_cat_cols[c][o]) == 0:
                # if teacher splits result in zero probability for a category value
                idx.append(np.random.randint(len(self._data)))
            else:
                idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return self._data[idx]

    def dim_cond_vec(self):
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id = self._discrete_column_matrix_st[condition_info["discrete_column_id"]
                                             ] + condition_info["value_id"]
        vec[:, id] = 1
        return vec
