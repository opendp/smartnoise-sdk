from typing import List, Union
import warnings
import numpy as np
from snsql.metadata import Metadata
from snsql.sql.odometer import OdometerHeterogeneous
from snsql.sql.privacy import Privacy, Stat

from snsql.sql.reader.base import SqlReader
from .dpsu import run_dpsu
from .private_rewriter import Rewriter
from .parse import QueryParser
from .reader import PandasReader
from .reader.base import SortKey

from snsql._ast.ast import Query, Top
from snsql._ast.expressions import sql as ast
from snsql._ast.expressions.date import parse_datetime
from snsql.reader import Reader

from ._mechanisms import *

import itertools

class PrivateReader(Reader):
    """Executes SQL queries against tabular data sources and returns differentially private results.

    PrivateReader should be created using the `from_connection` method.
    """
    def __init__(
        self,
        reader,
        metadata,
        privacy=None
    ):
        """Create a new private reader.  Do not use the constructor directly;
            use the from_connection factory method.

            :param metadata: The Metadata object with information about all tables referenced in this query
            :param reader: The data reader to wrap, such as a SqlServerReader, PandasReader, or SparkReader
                The PrivateReader intercepts queries to the underlying reader and ensures differential privacy.
            :param epsilon_per_column: The privacy budget to spend for each column in the query (deprecated)
            :param delta: The delta privacy parameter (deprecated)
            :param privacy: Pass epsilon and delta
        """
        if isinstance(reader, Reader):
            self.reader = reader
        else:
            raise ValueError("Parameter reader must be of type Reader")
        self.metadata = Metadata.from_(metadata)
        self.rewriter = Rewriter(metadata)
        self._options = PrivateReaderOptions()

        if privacy:
            self.privacy = privacy
        else:
            raise ValueError("Must pass in a Privacy object with privacy parameters.")
        
        self.odometer = OdometerHeterogeneous(self.privacy)

        self._refresh_options()
        self._warn_mechanisms()

    @classmethod
    def from_connection(cls, conn, *ignore, privacy, metadata, engine=None, **kwargs):
        """Create a private reader over an established SQL connection.  If `engine` is not
        passed in, the engine will be automatically detected.

        :param conn: An established database connection.  Can be pyodbc, psycopg2, SparkSession, Pandas DataFrame, or Presto.
        :param privacy:  A Privacy object with epsilon, delta, and other privacy properties.  Keyword-only.
        :param metadata: The metadata describing the database.  `Metadata documentation is here <https://docs.smartnoise.org/en/stable/sql/metadata.html>`_.  Keyword-only.
        :param engine: Optional keyword-only argument that can be used to specify engine-specific rules if automatic detection fails.  This should only be necessary when using an uncommon database or middleware.
        :returns: A `PrivateReader` object initialized to process queries against the supplied connection, using the supplied `Privacy` properties.

        .. code-block:: python
        
            privacy = Privacy(epsilon=1.0, delta=1/1000)
            metadata = 'datasets/PUMS.yaml'
            pums = pd.read_csv('datasets/PUMS.csv')
            reader = PrivateReader.from_connection(pums, privacy=privacy, metadata=metadata)
        """
        _reader = SqlReader.from_connection(conn, engine=engine, metadata=metadata, **kwargs)
        return cls(_reader, metadata, privacy=privacy)

    @property
    def engine(self) -> str:
        """The engine being used by this private reader.

        .. code-block:: python

            df = pd.read_csv('datasets/PUMS.csv')
            reader = from_connection(df, metadata=metadata, privacy=privacy)
            assert(reader.engine == 'pandas')
        """
        return self.reader.engine

    def _refresh_options(self):
        self.rewriter = Rewriter(self.metadata, privacy=self.privacy)
        self.metadata.compare = self.reader.compare
        tables = self.metadata.tables()
        self._options.row_privacy = any([t.row_privacy for t in tables])
        self._options.censor_dims = not any([not t.censor_dims for t in tables])
        self._options.reservoir_sample = any([t.sample_max_ids for t in tables])
        self._options.clamp_counts = any([t.clamp_counts for t in tables])
        self._options.max_contrib = max([t.max_ids for t in tables])
        self._options.use_dpsu = any([t.use_dpsu for t in tables])
        self._options.clamp_columns = any([t.clamp_columns for t in tables])

        self.rewriter.options.row_privacy = self._options.row_privacy
        self.rewriter.options.reservoir_sample = self._options.reservoir_sample
        self.rewriter.options.clamp_columns = self._options.clamp_columns
        self.rewriter.options.max_contrib = self._options.max_contrib
        self.rewriter.options.censor_dims = self._options.censor_dims

    def _warn_mechanisms(self):
        """
        Warn if any of the current settings could result in unsafe floating point mechanisms.
        """
        if self._options.censor_dims:
            warnings.warn(
f"""Dimension censoring is enabled, with {self.privacy.mechanisms.map[Stat.threshold]} as the thresholding mechanism. 
This is an unsafe floating point mechanism.  Counts used for censoring will be revealed in any queries that request COUNT DISTINCT(person), 
leading to potential privacy leaks. If your query workload needs to reveal distinct counts of individuals, consider doing the dimension
censoring as a preprocessing step.  See the documentation for more information."""
            )

        mechs = self.privacy.mechanisms
        tables = self.metadata.tables()
        floats = []
        large_ints = []
        large = mechs.large
        for table in tables:
            for col in table.columns():
                if col.typename() == 'float':
                    floats.append(col.name)
                elif col.typename() == 'int' and not col.unbounded:
                    if col.sensitivity and col.sensitivity >= large:
                        large_ints.append(col.name)
                    elif col.upper - col.lower >= large:
                        large_ints.append(col.name)
        if floats:
            warnings.warn(
f"""The following columns are of type float: {', '.join(floats)}. 
summary statistics over floats will use {mechs.map[Stat.sum_float]}, which is not floating-point safe, 
This could lead to privacy leaks."""
            )
    def _grouping_columns(self, query: Query):
        """
        Return a vector of boolean corresponding to the columns of the
        query, indicating which are grouping columns.
        """
        syms = query._select_symbols
        if query.agg is not None:
            group_keys = [
                ge.expression.name if hasattr(ge.expression, "name") else None
                for ge in query.agg.groupingExpressions
            ]
        else:
            group_keys = []
        return [colname in group_keys for colname in [s.name for s in syms]]
    def _aggregated_columns(self, query: Query):
        """
        Return a vector of boolean corresponding to columns of the
        query, indicating if the column is randomized
        """
        group_key = self._grouping_columns(query)
        agg = [s if s.expression.xpath("//AggFunction") else None for s in query._select_symbols]
        return [True if s and not g else False for s, g in zip(agg, group_key)]
    def _get_simple_accuracy(self, *ignore, query: Query, subquery: Query, alpha: float, **kwargs):
        """
        Return accuracy for each column in column order.  Currently only applies
        to simple aggregates, COUNT and SUM.  All other columns return None
        """
        agg = self._aggregated_columns(query)
        has_sens = [True if s.expression.sensitivity() else False for s in query._select_symbols]
        simple = [a and h for a, h in zip(agg, has_sens)]
        exprs = [ne.expression if simp else None for simp, ne in zip(simple, query.select.namedExpressions)]
        sources = [col.xpath_first("//Column") if col else None for col in exprs]
        col_names = [source.name if source else None for source in sources]
        mech_map = self._get_mechanism_map(subquery)
        mechs = [mech_map[name] if name and name in mech_map else None for name in col_names]
        accuracy = [mech.accuracy(alpha) if mech else None for mech in mechs]
        return accuracy
    def get_simple_accuracy(self, query_string: str, alpha: float):
        """
        Return accuracy for each alpha and each mechanism in column order.
        Columns with no mechanism application return None.  Returns accuracy
        without running the query.

        :param query_string: The SQL query
        :param alpha: The desired accuracy alpha.  For example, alpha of 0.05 will
            return a 95% interval.

        .. code-block:: python

            reader = from_df(df, metadata=metadata, privacy=privacy)
            query = 'SELECT COUNT(*) AS n, SUM(age) AS age FROM PUMS.PUMS GROUP BY income'

            accuracy = reader.get_simple_accuracy(query, 0.05)

            print(f'For 95% of query executions, n will be within +/- {accuracy[0]} of true value')
            print(f'For 95% of query executions, age will be within +/- {accuracy[1]} of true value')
        """
        subquery, query = self._rewrite(query_string)
        return self._get_simple_accuracy(query=query, subquery=subquery, alpha=alpha)
    
    def _get_mechanism_costs(self, query_string):
        """
        Return epsilon, delta cost for each mechanism in column order.
        Columns with no mechanism application return None.
        """
        self._refresh_options()
        subquery, _ = self._rewrite(query_string)
        mechs = self._get_mechanisms(subquery)
        return [(mech.epsilon, mech.delta) if mech else None for mech in mechs]
    
    def get_privacy_cost(self, query_strings: Union[str, List[str]]):
        """Estimates the epsilon and delta cost for running the given query.
        Privacy cost is returned without running the query or incrementing the odometer.

        :param query_string: The query string or strings to analyze
        :returns: A tuple of (epsilon, delta) estimating total privacy cost for
            running this query or queries.

        .. code-block:: python

            # metadata specifies censor_dims: False
            privacy = Privacy(epsilon=0.1, delta=1/1000)
            reader = from_df(df, metadata=metadata, privacy=privacy)

            query = 'SELECT AVG(age) FROM PUMS.PUMS GROUP BY educ'
            eps_cost, delta_cost = reader.get_privacy_cost(query)

            # will be ~0.2 epsilon, since AVG computed from SUM and COUNT
            print(f'Total epsilon spent will be {eps_cost}')

            query = 'SELECT SUM(age), COUNT(age), AVG(age) FROM PUMS.PUMS GROUP BY educ'
            eps_cost, delta_cost = reader.get_privacy_cost(query)

            # will be ~0.2 epsilon, since noisy SUM and COUNT are re-used
            print(f'Total epsilon spent will be {eps_cost}')

            query = 'SELECT COUNT(*), AVG(age) FROM PUMS.PUMS GROUP BY educ'
            eps_cost, delta_cost = reader.get_privacy_cost(query)

            # will be ~0.3 epsilon, since COUNT(*) and COUNT(age) can be different
            print(f'Total epsilon spent will be {eps_cost}')

        """
        odo = OdometerHeterogeneous(self.privacy)
        if not isinstance(query_strings, list):
            query_strings = [query_strings]
        for query_string in query_strings:
            costs = self._get_mechanism_costs(query_string)
            costs = [cost for cost in costs if cost]
            for epsilon, delta in costs:
                odo.spend(Privacy(epsilon=epsilon, delta=delta))
        return odo.spent

    def parse_query_string(self, query_string) -> Query:
        """Parse a query string, returning an AST `Query` object.

        .. code-block:: python

            reader = from_connection(db, metadata=metadata, privacy=privacy)
            query_string = 'SELECT STDDEV(age) AS age FROM PUMS.PUMS'
            query = reader.parse_query_string(query_string)
            age_node = query.xpath_first("//NamedExpression[@name='age']")
            dot = age_node.visualize() # visualize the formula in the AST
            dot.render('age', view=True, cleanup=True)

        """
        queries = QueryParser(self.metadata).queries(query_string)
        if len(queries) > 1:
            raise ValueError("Too many queries provided.  We can only execute one query at a time.")
        elif len(queries) == 0:
            return []
        return queries[0]

    def _rewrite(self, query_string):
        if not isinstance(query_string, str):
            raise ValueError("Please pass a query string to _rewrite()")
        query = self.parse_query_string(query_string)
        return self._rewrite_ast(query)

    def _rewrite_ast(self, query):
        if isinstance(query, str):
            raise ValueError("Please pass a Query AST object to _rewrite_ast()")
        query_max_contrib = query.max_ids
        if self._options.max_contrib is None or self._options.max_contrib > query_max_contrib:
            self._options.max_contrib = query_max_contrib

        self._refresh_options()
        query = self.rewriter.query(query)
        query.compare = self.reader.compare
        subquery = query.source.relations[0].primary.query
        subquery.compare = self.reader.compare
        return (subquery, query)

    def _get_reader(self, query_ast):
        if (
            query_ast.agg is not None
            and self._options.use_dpsu
            and isinstance(self.reader, PandasReader)
        ):
            query = str(query_ast)
            dpsu_df = run_dpsu(self.metadata, self.reader.df, query, epsilon=1.0)
            return PandasReader(dpsu_df, self.metadata)
        else:
            return self.reader
    def _get_mechanism_map(self, subquery: Query):
        """
        Returns a dictionary keyed by column name, with the instance of the
        mechanism used to randomize that column.
        """
        colnames = [s.name for s in subquery._select_symbols]
        mechs = self._get_mechanisms(subquery)
        mech_map = {}
        for name, mech in zip(colnames, mechs):
            if mech and name not in mech_map:
                mech_map[name] = mech
        return mech_map

    def _get_keycount_position(self, subquery: Query):
        """
        Returns the column index of the column that serves as the
        key count for tau thresholding.  Returns None if no keycount
        """
        is_key_count = [s.is_key_count for s in subquery._select_symbols]
        if any(is_key_count):
            return is_key_count.index(True)
        else:
            return None

    def _get_mechanisms(self, subquery: Query):
        max_contrib = self._options.max_contrib if self._options.max_contrib is not None else 1
        assert(subquery.max_ids == max_contrib)

        return [s.mechanism for s in subquery._select_symbols]

    def _check_pre_aggregated_columns(self, pre_aggregated, subquery: Query):
        """
        Checks to make sure the pre_aggregated iterable matches what would be
        expected if the generated subquery were executed.

        :param pre_aggregated: pre-aggregated values as would have been returned by
            executing the subquery.
        :param subquery: the subquery's AST, used to determine the column names and types.
        :returns: raises an error if the pre_aggregated shape do not match the expected shape.
            Otherwise, returns the pre_aggregated values suitable for further processing.
        """
        subquery_colnames = [s.name.split('_alias_')[0] for s in subquery._select_symbols]

        def normalize_colname(colname):
            # modify column names to make comparisons more reliable
            colname = colname.lower().replace(' ', '')
            colname = colname.split('_alias_')[0]
            colname = colname.replace('*', '').replace('(', '_').replace(')', '')
            return colname

        def check_colnames(colnames):
            if len(colnames) != len(subquery_colnames):
                raise ValueError(f"pre_aggregated has wrong number of columns, expected [{','.join(subquery_colnames)}], got [{','.join(colnames)}]")
            if not all([isinstance(c, str) for c in colnames]):
                raise ValueError(f"pre_aggregated column names must be strings, got {colnames}")
            colnames = [normalize_colname(c) for c in colnames]
            if not all([c == normalize_colname(s) for c, s in zip(colnames, subquery_colnames)]):
                raise ValueError(f"pre_aggregated column names must match subquery column names and order, expected [{','.join(subquery_colnames)}], got [{','.join(colnames)}]")

        if isinstance(pre_aggregated, str):
            raise ValueError("pre_aggregated must be a list of records")
        if isinstance(pre_aggregated, list):
            colnames = pre_aggregated[0]
            check_colnames(colnames)
        elif isinstance(pre_aggregated, np.ndarray):
            pass # ndarray does not have column names
        else:
            agg_mod = pre_aggregated.__class__.__module__
            agg_class = pre_aggregated.__class__.__name__
            if (
                agg_mod == 'pandas.core.frame' and
                agg_class == 'DataFrame'
            ):
                colnames = pre_aggregated.columns
                check_colnames(colnames)
                pre_aggregated = pre_aggregated.to_numpy()
            elif (
                agg_mod == 'pyspark.sql.dataframe' and
                agg_class == 'DataFrame'
            ):
                colnames = pre_aggregated.columns
                check_colnames(colnames)
            elif hasattr(pre_aggregated, 'map'):
                pass # RDD does not have column names
            else:
                raise ValueError("pre_aggregated must be a list of records")
        return pre_aggregated

    def execute_with_accuracy(self, query_string:str):
        """Executes a private SQL query, returning accuracy bounds for each column 
        and row.  This should only be used if you need analytic bounds for statistics
        where the bounds change based on partition size, such as AVG and VARIANCE.
        In cases where simple statistics such as COUNT and SUM are used, ``get_simple_accuracy``
        is recommended.  The analytic bounds for AVG and VARIANCE can be quite wide,
        so it's better to determine accuracy through simulation, whenever that's an option.

        Executes query and advances privacy odometer.  Returns accuracies for multiple alphas,
        using ``alphas`` property on the ``Privacy`` object that was passed in when the reader
        was instantiated.

        Note that the tuple format of ``execute_with_accuracy`` is not interchangeable with ``execute``,
        because the accuracy tuples need to be nested in the output rows to allow
        streamed processing.

        :param query_string: The query to execute.
        :returns: A tuple with a dataframe showing row results, and a nested
            tuple with a dataframe for each set of accuracies.  The accuracy
            dataframes will have the same number of rows and columns as the
            result dataframe.

        .. code-block:: python

            # alphas for 95% and 99% intervals
            privacy = Privacy(epsilon=0.1, delta=1/1000, alphas=[0.05, 0.01])
            reader = from_connection(db, metadata=metadata, privacy=privacy)            
            query = 'SELECT educ, AVG(age) AS age FROM PUMS.PUMS GROUP BY educ'

            res = reader.execute_with_accuracy(query)

            age_col = 2
            for row, accuracies in res:
                acc95, acc99 = accuracies
                print(f'Noisy average is {row[age_col]} with 95% +/- {acc95[age_col]} and 99% +/- {acc99[age_col]}')

        """
        return self.execute(query_string, accuracy=True)

    def execute_with_accuracy_df(self, query_string:str, *ignore):
        """Executes a private SQL query, returning accuracy bounds for each column 
        and row.  This should only be used if you need analytic bounds for statistics
        where the bounds change based on partition size, such as AVG and VARIANCE.
        In cases where simple statistics such as COUNT and SUM are used, ``get_simple_accuracy``
        is recommended.  The analytic bounds for AVG and VARIANCE can be quite wide,
        so it's better to determine accuracy through simulation, whenever that's an option.

        Executes query and advances privacy odometer.  Returns accuracies for multiple alphas,
        using ``alphas`` property on the ``Privacy`` object that was passed in when the reader
        was instantiated.

        Note that the tuple format of ``execute_with_accuracy_df`` is not interchangeable with 
        ``execute``, because the accuracy tuples need to be nested in the output rows to allow
        streamed processing.

        :param query_string: The query to execute.
        :returns: A list of tuples, with each item in the list representing a row.
            each row has a tuple of the result values, and a nested tuple with
            each of the column accuracies for that row, for each alpha.

        .. code-block:: python

            # alphas for 95% and 99% intervals
            privacy = Privacy(epsilon=0.1, delta=1/1000, alphas=[0.05, 0.01])
            reader = from_connection(db, metadata=metadata, privacy=privacy)            
            query = 'SELECT educ, AVG(age) AS age FROM PUMS.PUMS GROUP BY educ'

            res (acc95, acc99) = reader.execute_with_accuracy_df(query)

            print(res)
            print(acc95)
            print(acc99)
        """
        return self.execute_df(query_string, accuracy=True)

    def execute(self, query_string, accuracy:bool=False, *ignore, pre_aggregated=None, postprocess:bool=True):
        """Executes a query and returns a recordset that is differentially private.

        Follows ODBC and DB_API convention of consuming query as a string and returning
        recordset as tuples.  This is useful for cases where existing DB_API clients
        want to swap out API calls with minimal changes.

        :param query_string: A query string in SQL syntax        
        :param pre_aggregated: By default, `execute` will use the underlying database engine to compute exact aggregates.  To use exact aggregates from a different source, pass in the exact aggregates here as an iterable of tuples.
        :param postprocess: If False, the intermediate result, immediately after adding noise and censoring dimensions, will be returned.  All post-processing that does not impact privacy, such as clamping negative counts, LIMIT, HAVING, and ORDER BY, will be skipped.
        :return: A recordset structured as an array of tuples, where each tuple
         represents a row, and each item in the tuple is typed.  The first row will
         contain column names.

        .. code-block:: python
                
            result = reader.execute('SELECT sex, AVG(age) AS age FROM PUMS.PUMS GROUP BY sex')

        """
        query = self.parse_query_string(query_string)
        return self._execute_ast(
            query, 
            accuracy=accuracy, 
            pre_aggregated=pre_aggregated, 
            postprocess=postprocess
        )

    def _execute_ast(self, query, *ignore, accuracy:bool=False, pre_aggregated=None, postprocess=True):
        if isinstance(query, str):
            raise ValueError("Please pass AST to _execute_ast.")

        subquery, query = self._rewrite_ast(query)

        if pre_aggregated is not None:
            exact_aggregates = self._check_pre_aggregated_columns(pre_aggregated, subquery)
        else:
            exact_aggregates = self._get_reader(subquery)._execute_ast(subquery)

        _accuracy = None
        if accuracy:
            raise NotImplementedError("Simple accuracy has been removed.  Please see documentation for information on estimating accuracy.")

        syms = subquery._select_symbols
        source_col_names = [s.name for s in syms]

        # tell which are counts, in column order
        is_count = [s.expression.is_count for s in syms]

        # get a list of mechanisms in column order
        mechs = self._get_mechanisms(subquery)
        check_sens = [m for m in mechs if m]
        if any([m.sensitivity is np.inf for m in check_sens]):
            raise ValueError(f"Attempting to query an unbounded column")

        kc_pos = self._get_keycount_position(subquery)

        def randomize_row_values(row_in):
            row = [v for v in row_in]
            # set null to 0 before adding noise
            for idx in range(len(row)):
                if mechs[idx] and row[idx] is None:
                    row[idx] = 0.0
            # call all mechanisms to add noise
            return [
                mech.release([v])[0] if mech is not None else v
                for mech, v in zip(mechs, row)
            ]

        if hasattr(exact_aggregates, "rdd"):
            # it's a dataframe
            out = exact_aggregates.rdd.map(randomize_row_values)
        elif hasattr(exact_aggregates, "map"):
            # it's an RDD
            out = exact_aggregates.map(randomize_row_values)
        elif isinstance(exact_aggregates, list):
            out = map(randomize_row_values, exact_aggregates[1:])
        elif isinstance(exact_aggregates, np.ndarray):
            out = map(randomize_row_values, exact_aggregates)
        else:
            raise ValueError("Unexpected type for exact_aggregates")

        # censor infrequent dimensions
        if self._options.censor_dims:
            if kc_pos is None:
                raise ValueError("Query needs a key count column to censor dimensions")
            else:
                thresh_mech = mechs[kc_pos]
                self.tau = thresh_mech.threshold
            if hasattr(out, "filter"):
                # it's an RDD
                tau = self.tau
                out = out.filter(lambda row: row[kc_pos] > tau)
            else:
                out = filter(lambda row: row[kc_pos] > self.tau, out)

        if not postprocess:
            return out

        def process_clamp_counts(row_in):
            # clamp counts to be non-negative
            row = [v for v in row_in]
            for idx in range(len(row)):
                if is_count[idx] and row[idx] < 0:
                    row[idx] = 0
            return row

        clamp_counts = self._options.clamp_counts
        if clamp_counts:
            if hasattr(out, "rdd"):
                # it's a dataframe
                out = out.rdd.map(process_clamp_counts)
            elif hasattr(out, "map"):
                # it's an RDD
                out = out.map(process_clamp_counts)
            else:
                out = map(process_clamp_counts, out)

        # get column information for outer query
        out_syms = query._select_symbols
        out_types = [s.expression.type() for s in out_syms]
        out_col_names = [s.name for s in out_syms]

        def convert(val, type):
            if val is None:
                return None # all columns are nullable
            if type == "string" or type == "unknown":
                return str(val)
            elif type == "int":
                return int(float(str(val).replace('"', "").replace("'", "")))
            elif type == "float":
                return float(str(val).replace('"', "").replace("'", ""))
            elif type == "boolean":
                if isinstance(val, int):
                    return val != 0
                else:
                    return bool(str(val).replace('"', "").replace("'", ""))
            elif type == "datetime":
                v = parse_datetime(val)
                if v is None:
                    raise ValueError(f"Could not parse datetime: {val}")
                return v
            else:
                raise ValueError("Can't convert type " + type)
        
        alphas = [alpha for alpha in self.privacy.alphas]

        def process_out_row(row):
            bindings = dict((name.lower(), val) for name, val in zip(source_col_names, row))
            out_row = [c.expression.evaluate(bindings) for c in query.select.namedExpressions]
            try:
                out_row =[convert(val, type) for val, type in zip(out_row, out_types)]
            except Exception as e:
                raise ValueError(
                    f"Error converting output row: {e}\n"
                    f"Expecting types {out_types}"
                )

            # compute accuracies
            if accuracy == True and alphas:
                accuracies = [_accuracy.accuracy(row=list(row), alpha=alpha) for alpha in alphas]
                return tuple([out_row, accuracies])
            else:
                return tuple([out_row, []])

        if hasattr(out, "map"):
            # it's an RDD
            out = out.map(process_out_row)
        else:
            out = map(process_out_row, out)

        def filter_aggregate(row, condition):
            bindings = dict((name.lower(), val) for name, val in zip(out_col_names, row[0]))
            keep = condition.evaluate(bindings)
            return keep

        if query.having is not None:
            condition = query.having.condition
            if hasattr(out, "filter"):
                # it's an RDD
                out = out.filter(lambda row: filter_aggregate(row, condition))
            else:
                out = filter(lambda row: filter_aggregate(row, condition), out)

        # sort it if necessary
        if query.order is not None:
            sort_fields = []
            for si in query.order.sortItems:
                if type(si.expression) is not ast.Column:
                    raise ValueError("We only know how to sort by column names right now")
                colname = si.expression.name.lower()
                if colname not in out_col_names:
                    raise ValueError(
                        "Can't sort by {0}, because it's not in output columns: {1}".format(
                            colname, out_col_names
                        )
                    )
                colidx = out_col_names.index(colname)
                desc = False
                if si.order is not None and si.order.lower() == "desc":
                    desc = True
                if desc and not (out_types[colidx] in ["int", "float", "boolean", "datetime"]):
                    raise ValueError("We don't know how to sort descending by " + out_types[colidx])
                sf = (desc, colidx)
                sort_fields.append(sf)

            def sort_func(row):
                # use index 0, since index 1 is accuracy
                return SortKey(row[0], sort_fields)
                
            if hasattr(out, "sortBy"):
                out = out.sortBy(sort_func)
            else:
                out = sorted(out, key=sort_func)

        # check for LIMIT or TOP
        limit_rows = None
        if query.limit is not None:
            if query.select.quantifier is not None:
                raise ValueError("Query cannot have both LIMIT and TOP set")
            limit_rows = query.limit.n
        elif query.select.quantifier is not None and isinstance(query.select.quantifier, Top):
            limit_rows = query.select.quantifier.n
        if limit_rows is not None:
            if hasattr(out, "rdd"):
                # it's a dataframe
                out = out.limit(limit_rows)
            elif hasattr(out, "map"):
                # it's an RDD
                out = out.take(limit_rows)
            else:
                out = itertools.islice(out, limit_rows)


        # drop empty accuracy if no accuracy requested
        def drop_accuracy(row):
            return row[0]
        if accuracy == False:
            if hasattr(out, "rdd"):
                # it's a dataframe
                out = out.rdd.map(drop_accuracy)
            elif hasattr(out, "map"):
                # it's an RDD
                out = out.map(drop_accuracy)
            else:
                out = map(drop_accuracy, out)

        # increment odometer
        for mech in mechs:
            if mech:
                self.odometer.spend(Privacy(epsilon=mech.epsilon, delta=mech.delta))

        # output it
        if accuracy == False and hasattr(out, "toDF"):
            # Pipeline RDD
            if not out.isEmpty():
                return out.toDF(out_col_names)
            else:
                return out
        elif hasattr(out, "map"):
            # Bare RDD
            return out
        else:
            row0 = [out_col_names]
            if accuracy == True:
                row0 = [[out_col_names, [[col_name+'_' + str(1-alpha).replace('0.', '') for col_name in out_col_names] for alpha in self.privacy.alphas ]]]
            out_rows = row0 + list(out)
            return out_rows

    def _execute_ast_df(self, query):
        return self._to_df(self._execute_ast(query))


class PrivateReaderOptions:
    """Options that control privacy behavior"""

    def __init__(
        self,
        censor_dims=True,
        clamp_counts=True,
        reservoir_sample=True,
        clamp_columns=True,
        row_privacy=False,
        max_contrib=None,
        use_dpsu=True,
    ):
        """Initialize with options.

        :param censor_dims: boolean, set to False if you know that small dimensions cannot expose privacy
        :param clamp_counts: boolean, set to False to allow noisy counts to be negative
        :param reservoir_sample: boolean, set to False if the data collection will never have more than max_contrib record per individual
        :param clamp_columns: boolean, set to False to allow values that exceed lower and higher limit specified in metadata.  May impact privacy
        :param row_privacy: boolean, True if each row is a separate individual
        :param max_contrib: int, set to override the metadata-supplied limit of per-user
          contribution.  May only revise down; metadata takes precedence if limit is smaller.
        :param use_dpsu: boolean, set to False if you want to use DPSU for histogram queries
        """

        self.censor_dims = censor_dims
        self.clamp_counts = clamp_counts
        self.reservoir_sample = reservoir_sample
        self.clamp_columns = clamp_columns
        self.row_privacy = row_privacy
        self.max_contrib = max_contrib
        self.use_dpsu = use_dpsu
