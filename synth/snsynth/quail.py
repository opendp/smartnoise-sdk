import logging
import warnings

import numpy as np
import pandas as pd

from snsynth.base import Synthesizer

logger = logging.getLogger(__name__)


class QUAILSynthesizer(Synthesizer):
    """
    Quailified Architecture to Improve Labeling.
    Divide epsilon in a known classification task
    between a differentially private synthesizer and
    classifier. Train DP classifier on real, fit DP synthesizer
    to features (excluding the target label),
    and use synthetic data from the DP synthesizer with
    the DP classifier to create artificial labels. Produces
    complete synthetic data.

    More information here:
    Differentially Private Synthetic Data: Applied Evaluations and Enhancements
    https://arxiv.org/abs/2011.05537

    :param epsilon: Total epsilon used across the DP Synthesizer and DP Classifier
    :type epsilon: float
    :param dp_synthesizer: A function that returns an instance of a DP Synthesizer
        for a specified epsilon value
    :type dp_synthesizer: function (epsilon) -> SDGYMBaseSynthesizer
    :param dp_classifier: A function that returns an instance of a DP Classifier
        for a specified epsilon value
    :type dp_classifier: function (epsilon) -> classifier
    :param target: The column name of the target column
    :type target: str
    :param test_size: Percent of the data that should be used for the test set,
        defaults to 0.2
    :type test_size: float, optional
    :param seed: Seed for controlling randomness for testing, defaults to None
    :type seed: int, optional
    :param eps_split: Percent of epsilon used for the classifier.
        1 - eps_split is used for the Synthesizer., defaults to 0.9
    :type eps_split: float, optional
    """
    def __init__(
        self,
        epsilon,
        dp_synthesizer,
        dp_classifier,
        target,
        test_size=0.2,
        seed=None,
        eps_split=0.9,
    ):
        """
        Quailified Architecture to Improve Labeling.
        Divide epsilon in a known classification task
        between a differentially private synthesizer and
        classifier. Train DP classifier on real, fit DP synthesizer
        to features (excluding the target label),
        and use synthetic data from the DP synthesizer with
        the DP classifier to create artificial labels. Produces
        complete synthetic data.

        More information here:
        Differentially Private Synthetic Data: Applied Evaluations and Enhancements
        https://arxiv.org/abs/2011.05537

        :param epsilon: Total epsilon used across the DP Synthesizer and DP Classifier
        :type epsilon: float
        :param dp_synthesizer: A function that returns an instance of a DP Synthesizer
            for a specified epsilon value
        :type dp_synthesizer: function (epsilon) -> SDGYMBaseSynthesizer
        :param dp_classifier: A function that returns an instance of a DP Classifier
            for a specified epsilon value
        :type dp_classifier: function (epsilon) -> classifier
        :param target: The column name of the target column
        :type target: str
        :param test_size: Percent of the data that should be used for the test set,
            defaults to 0.2
        :type test_size: float, optional
        :param seed: Seed for controlling randomness for testing, defaults to None
        :type seed: int, optional
        :param eps_split: Percent of epsilon used for the classifier.
            1 - eps_split is used for the Synthesizer., defaults to 0.9
        :type eps_split: float, optional
        """
        self.epsilon = epsilon
        self.eps_split = eps_split
        self.dp_synthesizer = dp_synthesizer
        self.dp_classifier = dp_classifier
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

    def fit(
        self,
        data,
        categorical_columns=tuple(),
        ordinal_columns=tuple(),
        transformer=None,
        continuous_columns=None,
        verbose=None,
        preprocessor_eps=0.0,
        nullable=False,
    ):
        """
        Takes a dataset and fits the synthesizer/learning model to it, using the epsilon split
        specified in the init.

        :param data: Data
        :type data: pd.DataFrame or np.array
        """
        if verbose is not None:
            warnings.warn("verbose is deprecated. Use logging.setLevel instead")

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score

        if isinstance(data, pd.DataFrame):
            self.pandas = True
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="ignore")
            self.data = data
            self.pd_cols = data.columns
            self.pd_index = data.index
        else:
            raise ("Only pandas dataframes for data as of now.")

        private_features = data.loc[:, data.columns != self.target]
        private_target = data.loc[:, data.columns == self.target]
        x_train, x_test, y_train, y_test = train_test_split(
            private_features,
            private_target,
            test_size=self.test_size,
            random_state=self.seed,
        )

        # Here we train a differentially private model on the real
        # data. We report on the accuracy for now to give a sense of
        # the upper bound on performance in the sampling step.
        self.private_model = self.dp_classifier(epsilon=(self.epsilon * self.eps_split))
        self.private_model.fit(x_train, y_train.values.ravel())
        predictions = self.private_model.predict(x_test)
        self.class_report = classification_report(
            np.ravel(y_test), predictions, labels=np.unique(predictions)
        )
        self.target_accuracy = accuracy_score(np.ravel(y_test), predictions)
        log_level = logger.level
        if verbose:
            log_level = logging.INFO

        logging.log(log_level, "Internal model report: ")
        logging.log(log_level, self.class_report)
        logging.log(log_level, self.target_accuracy)

        # We use the features in our synthesis.
        self.private_synth = self.dp_synthesizer(
            epsilon=(self.epsilon * (1 - self.eps_split))
        )
        self.private_synth.fit(
            data=private_features,
            categorical_columns=categorical_columns,
            ordinal_columns=ordinal_columns,
            transformer=transformer,
            continuous_columns=continuous_columns,
            preprocessor_eps=preprocessor_eps,
            nullable=nullable,
        )

        if hasattr(self.private_model, "coef_"):
            logging.log(log_level, self.private_model.coef_)

        if hasattr(self.private_model, "intercept_"):
            logging.log(log_level, self.private_model.intercept_)

        if hasattr(self.private_model, "classes_"):
            logging.log(log_level, self.private_model.classes_)

    def sample(self, samples):
        """
        Sample from the synthesizer model.

        :param samples: The number of samples to create
        :type samples: int
        :return: A dataframe of length samples
        :rtype: pd.Dataframe
        """
        sampled_features = self.private_synth.sample(samples)
        y_values = self.private_model.predict(sampled_features)

        sampled_features[self.target] = y_values
        return sampled_features
