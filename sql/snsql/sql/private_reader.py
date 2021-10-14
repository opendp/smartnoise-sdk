import logging
import numpy as np
from snsql.metadata import Metadata
from snsql.sql._mechanisms.accuracy import Accuracy
from snsql.sql.odometer import OdometerHeterogeneous
from snsql.sql.privacy import Privacy

from snsql.sql.reader.base import SqlReader
from .dpsu import run_dpsu
from .private_rewriter import Rewriter
from .parse import QueryParser
from .reader import PandasReader

from snsql._ast.ast import Query, Top
from snsql._ast.expressions import sql as ast
from snsql.reader import Reader

from ._mechanisms import *

module_logger = logging.getLogger(__name__)

import itertools

class PrivateReader(Reader):
    """Executes SQL queries against tabular data sources and returns differentially private results.

    PrivateReader should be created using the `from_connection` factory method.  For example,
    using pyodbc:
    
    .. code-block:: python

        conn = pyodbc.connect(dsn)
        meta = 'datasets/PUMS.yaml'
        privacy = Privacy(epsilon=0.1, delta=1/10000)
        reader = PrivateReader.from_connection(conn, metadata=meta, privacy=privacy)

        result = reader.execute('SELECT COUNT(*) AS n FROM PUMS.PUMS GROUP BY educ')

    or using a Pandas dataframe:

    .. code-block:: python

        csv = 'datasets/PUMS.csv'
        pums = pd.read_csv(csv)
        meta = 'datasets/PUMS.yaml'

        privacy = Privacy(epsilon=0.1, delta=1/10000)
        reader = PrivateReader.from_connection(pums, metadata=meta, privacy=privacy)

        result = reader.execute('SELECT COUNT(*) AS n FROM PUMS.PUMS GROUP BY educ')

    """

    def __init__(
        self,
        reader,
        metadata,
        epsilon_per_column=1.0,
        delta=10e-16,
        *ignore,
        privacy=None

    ):
        """Create a new private reader.

            :param metadata: The Metadata object with information about all tables referenced in this query
            :param reader: The data reader to wrap, such as a SqlServerReader, PandasReader, or SparkReader
                The PrivateReader intercepts queries to the underlying reader and ensures differential privacy.
            :param epsilon_per_column: The privacy budget to spend for each column in the query
            :param delta: The delta privacy parameter
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
            self.privacy = Privacy(epsilon=epsilon_per_column, delta=delta)
        
        self.odometer = OdometerHeterogeneous(self.privacy)

        self._refresh_options()

    @classmethod
    def from_connection(cls, conn, *ignore, privacy, metadata, engine=None, **kwargs):
        """Create a private reader over an established SQL connection.  If `engine` is not
        passed in, the engine will be automatically detected.

        :param conn: An established database connection.  Can be pyodbc, psycopg2, SparkSession, Pandas DataFrame, or Presto.
        :param privacy:  A Privacy object with epsilon, delta, and other privacy properties.  Keyword-only.
        :param metadata: The metadata describing the database.  `Metadata documentation is here <https://github.com/opendp/smartnoise-sdk/blob/new_opendp/sdk/Metadata.md>`_.  Keyword-only.
        :param engine: Optional keyword-only argument that can be used to specify engine-specific rules if automatic detection fails.  This should only be necessary when using an uncommon database or middleware.
        :returns: A `PrivateReader` object initialized to process queries against the supplied connection, using the supplied `Privacy` properties.
        """
        _reader = SqlReader.from_connection(conn, engine=engine, metadata=metadata, **kwargs)
        return PrivateReader(_reader, metadata, privacy=privacy)

    @property
    def engine(self) -> str:
        """The engine being used by this private reader.

            df = pd.read_csv('datasets/PUMS.csv')
            reader = PrivateReader.from_connection(df, metadata=meta, privacy=privacy)
            assert(reader.engine == 'pandas')
        """
        return self.reader.engine

    def _refresh_options(self):
        self.rewriter = Rewriter(self.metadata)
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

    def _grouping_columns(self, query: Query):
        """
        Return a vector of boolean corresponding to the columns of the
        query, indicating which are grouping columns.
        """
        syms = query.all_symbols()
        if query.agg is not None:
            group_keys = [
                ge.expression.name if hasattr(ge.expression, "name") else None
                for ge in query.agg.groupingExpressions
            ]
        else:
            group_keys = []
        return [colname in group_keys for colname in [s[0] for s in syms]]
    def _aggregated_columns(self, query: Query):
        """
        Return a vector of boolean corresponding to columns of the
        query, indicating if the column is randomized
        """
        group_key = self._grouping_columns(query)
        agg = [s if s[1].xpath("//AggFunction") else None for s in query.m_symbols]
        return [True if s and not g else False for s, g in zip(agg, group_key)]
    def _get_simple_accuracy(self, *ignore, query: Query, subquery: Query, alpha: float, **kwargs):
        """
        Return accuracy for each column in column order.  Currently only applies
        to simple aggregates, COUNT and SUM.  All other columns return None
        """
        agg = self._aggregated_columns(query)
        has_sens = [True if s[1].sensitivity() else False for s in query.m_symbols]
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
        Columns with no mechanism application return None.
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
    
    def get_privacy_cost(self, query_string):
        """Estimates the epsilon and delta cost for running the given query.
        """
        odo = OdometerHeterogeneous(self.privacy)
        costs = self._get_mechanism_costs(query_string)
        costs = [cost for cost in costs if cost]
        for epsilon, delta in costs:
            odo.spend(Privacy(epsilon=epsilon, delta=delta))
        return odo.spent

    def parse_query_string(self, query_string) -> Query:
        """Parse a query string using this `PrivateReader`'s metadata, returning a `Query` from the AST.

            reader = PrivateReader.from_connection(pums, metadata=meta, privacy=privacy)
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
        colnames = [s[0] for s in subquery.m_symbols]
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
        kc_pos = None
        syms = subquery.all_symbols()
        for idx in range(len(syms)):
            sname, _ = syms[idx]
            if sname == "keycount":
                kc_pos = idx
        return kc_pos

    def _get_mechanisms(self, subquery: Query):
        max_contrib = self._options.max_contrib if self._options.max_contrib is not None else 1

        syms = subquery.all_symbols()
        kc_pos = self._get_keycount_position(subquery)

        cols = [(s[1].sensitivity(), s[1].type(), s[1].is_count) for s in syms]

        is_group_key = self._grouping_columns(subquery)
        cols = zip(is_group_key, cols)
        cols = [(None if gk else s, t, c) for gk, (s, t, c) in cols]
        if any([s is np.inf for s, _, _ in cols]):
            raise ValueError(
                "Query is attempting to query an unbounded column that isn't part of the grouping key"
            )
        mechanisms = self.privacy.mechanisms
        epsilon = self.privacy.epsilon
        delta = self.privacy.delta
        mechs = []
        for idx in range(len(cols)):
            sensitivity, t, is_count = cols[idx]
            mech = None
            if t in ['int', 'float'] and sensitivity is not None:
                stat = 'count' if is_count else 'sum'
                if kc_pos is not None and idx == kc_pos:
                    stat = 'threshold'
                mech_class = mechanisms.get_mechanism(sensitivity, stat, t)
                mech = mech_class(epsilon, delta=delta, sensitivity=sensitivity, max_contrib=max_contrib)
                if kc_pos is not None and idx == kc_pos:
                    mech.delta = delta
            mechs.append(mech)
        return mechs

    def execute_with_accuracy(self, query_string:str):
        return self.execute(query_string, accuracy=True)

    def execute_with_accuracy_df(self, query_string:str, *ignore, privacy:bool=False):
        return self.execute_df(query_string, accuracy=True)


    def execute(self, query_string, accuracy:bool=False, *ignore, _pre_aggregated=None, _no_postprocess=None):
        """Executes a query and returns a recordset that is differentially private.

        Follows ODBC and DB_API convention of consuming query as a string and returning
        recordset as tuples.  This is useful for cases where existing DB_API clients
        want to swap out API calls with minimal changes.

        :param query_string: A query string in SQL syntax
        :return: A recordset structured as an array of tuples, where each tuple
         represents a row, and each item in the tuple is typed.  The first row should
         contain column names.
        """
        query = self.parse_query_string(query_string)
        return self._execute_ast(
            query, 
            accuracy=accuracy, 
            _pre_aggregated=_pre_aggregated, 
            _no_postprocess=_no_postprocess
        )

    def _execute_ast(self, query, *ignore, accuracy:bool=False, _pre_aggregated=None, _no_postprocess=None):
        if isinstance(query, str):
            raise ValueError("Please pass AST to _execute_ast.")

        subquery, query = self._rewrite_ast(query)

        if _pre_aggregated:
            exact_aggregates = _pre_aggregated
        else:
            exact_aggregates = self._get_reader(subquery)._execute_ast(subquery)

        _accuracy = None
        if accuracy:
            _accuracy = Accuracy(query, subquery, self.privacy)

        syms = subquery.all_symbols()
        source_col_names = [s[0] for s in syms]

        # tell which are counts, in column order
        is_count = [s[1].is_count for s in syms]

        # get a list of mechanisms in column order
        mechs = self._get_mechanisms(subquery)

        kc_pos = self._get_keycount_position(subquery)
        if kc_pos is not None:
            thresh_mech = mechs[kc_pos]
            self.tau = thresh_mech.threshold

        def randomize_row(row_in):
            row = [v for v in row_in]
            # set null to 0 before adding noise
            for idx in range(len(row)):
                #if sens[idx] is not None and row[idx] is None:
                if mechs[idx] and row[idx] is None:
                    row[idx] = 0.0
            # call all mechanisms to add noise
            return [
                mech.release([v])[0] if mech is not None else v
                for mech, v in zip(mechs, row)
            ]

        if hasattr(exact_aggregates, "rdd"):
            # it's a dataframe
            out = exact_aggregates.rdd.map(randomize_row)
        elif hasattr(exact_aggregates, "map"):
            # it's an RDD
            out = exact_aggregates.map(randomize_row)
        else:
            out = map(randomize_row, exact_aggregates[1:])

        if _no_postprocess:
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

        # censor infrequent dimensions
        if self._options.censor_dims:
            if kc_pos is None:
                raise ValueError("Query needs a key count column to censor dimensions")
            if hasattr(out, "filter"):
                # it's an RDD
                tau = self.tau
                out = out.filter(lambda row: row[kc_pos] > tau)
            else:
                out = filter(lambda row: row[kc_pos] > self.tau, out)

        # get column information for outer query
        out_syms = query.all_symbols()
        out_types = [s[1].type() for s in out_syms]
        out_col_names = [s[0] for s in out_syms]

        def convert(val, type):
            if type == "string" or type == "unknown":
                return str(val).replace('"', "").replace("'", "")
            elif type == "int":
                return int(float(str(val).replace('"', "").replace("'", "")))
            elif type == "float":
                return float(str(val).replace('"', "").replace("'", ""))
            elif type == "boolean":
                if isinstance(val, int):
                    return val != 0
                else:
                    return bool(str(val).replace('"', "").replace("'", ""))
            else:
                raise ValueError("Can't convert type " + type)
        
        alphas = [alpha for alpha in self.privacy.alphas]

        def process_out_row(row):
            bindings = dict((name.lower(), val) for name, val in zip(source_col_names, row))
            out_row = [c.expression.evaluate(bindings) for c in query.select.namedExpressions]
            out_row =[convert(val, type) for val, type in zip(out_row, out_types)]

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
                if desc and not (out_types[colidx] in ["int", "float", "boolean"]):
                    raise ValueError("We don't know how to sort descending by " + out_types[colidx])
                sf = (desc, colidx)
                sort_fields.append(sf)

            def sort_func(row):
                row_value, accuracies = row
                out_row_value = tuple(
                    [
                        row_value[idx]
                        if not desc
                        else not row_value[idx]
                        if out_types[idx] == "boolean"
                        else -row_value[idx]
                        for desc, idx in sort_fields
                    ]
                )
                return tuple([out_row_value, accuracies])
                
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
                if hasattr(exact_aggregates, "rdd"):
                    # it's a dataframe
                    out = exact_aggregates.limit(limit_rows)
                elif hasattr(exact_aggregates, "map"):
                    # it's an RDD
                    out = exact_aggregates.limit(limit_rows)
                else:
                    out = itertools.islice(out, limit_rows)

        # drop empty accuracy if no accuracy requested
        def drop_accuracy(row):
            return row[0]
        if accuracy == False:
            if hasattr(out, "map"):
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
            return out.toDF(out_col_names)
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
