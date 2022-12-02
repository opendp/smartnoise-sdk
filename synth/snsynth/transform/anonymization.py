from .base import ColumnTransformer
from .definitions import ColumnType
from faker import Faker


class AnonymizationTransformer(ColumnTransformer):
    """
    Transformer that can be used to anonymize personally identifiable information (PII) or other values.
    By default, the existing values are discarded during transformation and not used by a synthesizer.
    During inverse transformation new values will be generated according to the specified ``fake``.

    If ``fake_inbound`` is true, the new values will be injected during transformation and passed through on inverse.
    This might be useful for e.g. operation in a ChainTransformer.

    Beware that the provided ``fake`` is called once to verify that the provided (keyword) arguments are valid.

    :param fake: Text reference to Faker method (e.g. 'email') or custom callable
    :type fake: str or callable, required
    :param args: Arguments for the method
    :type args: args, optional
    :param faker_setup: Dictionary with keyword arguments for Faker initialization e.g. {'locale': 'de_DE'}
    :type faker_setup: dict, optional
    :param fake_inbound: Defaults to False.
    :type fake_inbound: bool, optional
    :param kwargs: Keyword arguments for the method
    :type kwargs: kwargs, optional
    """

    def __init__(self, fake, *args, faker_setup=None, fake_inbound=False, **kwargs):
        self.fake_inbound = fake_inbound
        super().__init__()

        if isinstance(fake, str):  # assume this references a Faker builtin
            fake = self._get_faker_builtin(fake, faker_setup, *args, **kwargs)

        self.fake = fake
        self.args = args
        self.kwargs = kwargs

        # verify that the provided arguments are valid
        try:
            self._generate_fake_data()
        except TypeError as e:
            raise ValueError(
                f"Provided arguments {args} and {kwargs} are invalid for `fake` {fake}"
            ) from e

    def _get_faker_builtin(self, fake, faker_setup, *args, **kwargs):
        """
        Creates a Faker instance and verifies that the given method is available.

        :param fake: Text reference to Faker method
        :type fake: str, required
        :param faker_setup: Dictionary with keyword arguments for initializing Faker e.g. {'locale': 'de_DE'}
        :type faker_setup: dict, optional
        :param args: Arguments for the method
        :type args: args, optional
        :param kwargs: Keyword arguments for the method
        :type kwargs: kwargs, optional
        :return: Actual Faker method
        :rtype: callable
        """
        # initialize Faker with provided setup or default
        if isinstance(faker_setup, dict):
            try:
                self.faker = Faker(**faker_setup)
            except Exception as e:
                raise ValueError(
                    f"Provided `faker_setup` {faker_setup} is invalid"
                ) from e
        else:
            self.faker = Faker()

        # verify that the provided fake is available
        try:
            fake_builtin = getattr(self.faker, fake)
        except AttributeError as e:
            raise ValueError(f"Provided `fake` {fake} is not available in Faker") from e

        return fake_builtin

    @property
    def output_type(self):
        return ColumnType.UNBOUNDED

    @property
    def cardinality(self):
        return [None]

    def _fit(self, _):
        pass

    def _clear_fit(self):
        self._fit_complete = True
        self.output_width = 1 if self.fake_inbound else 0

    def _generate_fake_data(self):
        return self.fake(*self.args, **self.kwargs)

    def _transform(self, _):
        if self.fake_inbound:
            return self._generate_fake_data()
        else:
            return None

    def _inverse_transform(self, val):
        if self.fake_inbound:
            return val
        else:
            return self._generate_fake_data()

    def transform(self, data, idx=None):
        if idx is None:
            return [self._transform(val) for val in data]
        else:
            return [row[:idx] + row[idx + 1 :] for row in data]

    def inverse_transform(self, data, idx=None):
        if idx is None:
            return [self._inverse_transform(val) for val in data]
        else:
            return [
                row[:idx] + (self._inverse_transform(None),) + row[idx:] for row in data
            ]
