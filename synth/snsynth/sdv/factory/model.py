"""Hierarchical Modeling Algorithms."""

import logging

import numpy as np
import pandas as pd

from snsynth.sdv.factory.base import BaseModel
from snsynth.sdv.tabular.copulas import GaussianCopula

LOGGER = logging.getLogger(__name__)


class FactoryModel(BaseModel):
    """Hierarchical Modeling Algorithm One.

    Args:
        metadata (dict, str or Metadata):
            Metadata dict, path to the metadata JSON file or Metadata instance itself.
        root_path (str or None):
            Path to the dataset directory. If ``None`` and metadata is
            a path, the metadata location is used. If ``None`` and
            metadata is a dict, the current working directory is used.
        model (type):
            Class of the ``copula`` to use. Defaults to
            ``sdv.models.copulas.GaussianCopula``.
        model_kwargs (dict):
            Keyword arguments to pass to the model. If the default model is used, this
            defaults to using a ``gaussian`` distribution and a ``categorical_fuzzy``
            transformer.
    """

    DEFAULT_MODEL = GaussianCopula
    DEFAULT_MODEL_KWARGS = {
        'default_distribution': 'gaussian',
        'categorical_transformer': 'label_encoding',
    }

    def __init__(self, metadata, root_path=None, model=None, model_kwargs=None):
        super().__init__(metadata, root_path)

        if model is None:
            model = self.DEFAULT_MODEL
            if model_kwargs is None:
                model_kwargs = self.DEFAULT_MODEL_KWARGS

        self._model = model
        self._model_kwargs = model_kwargs or {}
        self._models = {}
        self._table_sizes = {}
        self._max_child_rows = {}

    # ######## #
    # MODELING #
    # ######## #

    def _load_table(self, tables, table_name):
        """Load the specified table.

        Args:
            tables (dict or None):
                A dictionary mapping table name to table.
            table_name (str):
                The name of the desired table.

        Returns:
            pandas.DataFrame
        """
        if tables and table_name in tables:
            table = tables[table_name].copy()
        else:
            table = self.metadata.load_table(table_name)
            tables[table_name] = table

        return table

    def _prepare_for_modeling(self, table_data, table_name, primary_key):
        """Prepare the given table for modeling.

        In preparation for modeling a given table, do the following:
        - drop the primary key if exists
        - drop any other columns of type 'id'
        - add unknown fields to metadata as numerical fields,
          and fill missing values in those fields

        Args:
            table_data (pandas.DataFrame):
                The data of the desired table.
            table_name (str):
                The name of the table.
            primary_key (str):
                The name of the primary key column.

        Returns:
            (dict, dict):
                A tuple containing the table metadata to use for modeling, and
                the values of the id columns.
        """
        table_meta = self.metadata.get_table_meta(table_name)
        table_meta['name'] = table_name

        fields = table_meta['fields']

        if primary_key:
            table_meta['primary_key'] = None
            del table_meta['fields'][primary_key]

        keys = {}
        for name, field in list(fields.items()):
            if field['type'] == 'id':
                keys[name] = table_data.pop(name).values
                del fields[name]

        for column in table_data.columns:
            if column not in fields:
                fields[column] = {
                    'type': 'numerical',
                    'subtype': 'float'
                }

                column_data = table_data[column]
                if column_data.dtype in (np.int, np.float):
                    fill_value = column_data.mean()
                else:
                    fill_value = column_data.mode()[0]

                table_data[column] = table_data[column].fillna(fill_value)

        return table_meta, keys

    def _model_table(self, table_name, tables):
        """Model the indicated table and its children.

        Args:
            table_name (str):
                Name of the table to model.
            tables (dict):
                Dict of original tables.

        Returns:
            pandas.DataFrame:
                table data with the extensions created while modeling its children.
        """
        LOGGER.info('Modeling %s', table_name)

        table = self._load_table(tables, table_name)
        self._table_sizes[table_name] = len(table)

        primary_key = self.metadata.get_primary_key(table_name)
        if primary_key:
            table = table.set_index(primary_key)
            # NOTE: Smartnoise does not support factory tables, so
            # we do not extend our dataframe here with children

        table_meta, keys = self._prepare_for_modeling(table, table_name, primary_key)

        LOGGER.info('Fitting %s for table %s; shape: %s', self._model.__name__,
                    table_name, table.shape)
        model = self._model(**self._model_kwargs, table_metadata=table_meta)
        model.fit(table)
        self._models[table_name] = model

        if primary_key:
            table.reset_index(inplace=True)

        for name, values in keys.items():
            table[name] = values

        tables[table_name] = table

        return table

    def _fit(self, tables=None):
        """Fit this FactoryModel instance to the dataset data.

        Args:
            tables (dict):
                Dictionary with the table names as key and ``pandas.DataFrame`` instances as
                values.  If ``None`` is given, the tables will be loaded from the paths
                indicated in ``metadata``. Defaults to ``None``.
        """
        self.metadata.validate(tables)
        if tables:
            tables = tables.copy()
        else:
            tables = {}

        for table_name in self.metadata.get_tables():
            if not self.metadata.get_parents(table_name):
                self._model_table(table_name, tables)

        LOGGER.info('Modeling Complete')

    # ######## #
    # SAMPLING #
    # ######## #

    def _finalize(self, sampled_data):
        """Do the final touches to the generated data.

        This method reverts the previous transformations to go back
        to values in the original space and also adds the parent
        keys in case foreign key relationships exist between the tables.

        Args:
            sampled_data (dict):
                Generated data

        Return:
            pandas.DataFrame:
                Formatted synthesized data.
        """
        final_data = dict()
        for table_name, table_rows in sampled_data.items():
            dtypes = self.metadata.get_dtypes(table_name, ids=True)
            for name, dtype in dtypes.items():
                table_rows[name] = table_rows[name].dropna().astype(dtype)

            final_data[table_name] = table_rows[list(dtypes.keys())]

        return final_data

    def _sample_rows(self, model, table_name, num_rows=None):
        """Sample ``num_rows`` from ``model``.

        Args:
            model (copula.multivariate.base):
                Fitted model.
            table_name (str):
                Name of the table to sample from.
            num_rows (int):
                Number of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled rows, shape (, num_rows)
        """
        sampled = model.sample(num_rows)

        primary_key_name = self.metadata.get_primary_key(table_name)
        if primary_key_name:
            primary_key_values = self._get_primary_keys(table_name, len(sampled))
            sampled[primary_key_name] = primary_key_values

        return sampled

    def _sample_table(self, table_name, num_rows=None, sampled_data=None): # sample_children=True,
        """Sample a single table and optionally its children."""
        if sampled_data is None:
            sampled_data = {}

        if num_rows is None:
            num_rows = self._table_sizes[table_name]

        LOGGER.info('Sampling %s rows from table %s', num_rows, table_name)

        model = self._models[table_name]
        table_rows = self._sample_rows(model, table_name, num_rows)
        sampled_data[table_name] = table_rows

        # if sample_children:
        #     self._sample_children(table_name, sampled_data, table_rows)

        return sampled_data

    def _sample(self, table_name=None, num_rows=None): #, sample_children=True):
        """Sample the entire dataset.

        ``sample_all`` returns a dictionary with all the tables of the dataset sampled.
        The amount of rows sampled will depend from table to table, and is only guaranteed
        to match ``num_rows`` on tables without parents.

        This is because the children tables are created modelling the relation that they have
        with their parent tables, so its behavior may change from one table to another.

        Args:
            num_rows (int):
                Number of rows to be sampled on the first parent tables. If ``None``,
                sample the same number of rows as in the original tables.
            reset_primary_keys (bool):
                Whether or not reset the primary key generators.

        Returns:
            dict:
                A dictionary containing as keys the names of the tables and as values the
                sampled datatables as ``pandas.DataFrame``.

        Raises:
            NotFittedError:
                A ``NotFittedError`` is raised when the ``SDV`` instance has not been fitted yet.
        """
        if table_name:
            sampled_data = self._sample_table(table_name, num_rows) #, sample_children)
            sampled_data = self._finalize(sampled_data)
            # if sample_children:
            #     return sampled_data

            return sampled_data[table_name]

        sampled_data = dict()
        for table in self.metadata.get_tables():
            if not self.metadata.get_parents(table):
                self._sample_table(table, num_rows, sampled_data=sampled_data)

        return self._finalize(sampled_data)
