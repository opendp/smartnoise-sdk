import logging
import warnings
import math
import numpy as np
from .dpsu import run_dpsu
from .private_rewriter import Rewriter
from .parse import QueryParser
from .reader import PandasReader

from opendp.smartnoise._ast.ast import Top
from opendp.smartnoise._ast.expressions import sql as ast
from opendp.smartnoise.reader import Reader

from ._mechanisms.gaussian import Gaussian
from opendp.smartnoise.report import Interval, Intervals, Result

module_logger = logging.getLogger(__name__)

import itertools


class PrivateReader(Reader):
    """Executes SQL queries against tabular data sources and returns differentially private results
    """

    def __init__(
        self,
        reader,
        metadata,
        epsilon_per_column,
        delta=10e-16,
        interval_widths=None,
        options=None,
        epsilon=None,
    ):
        """Create a new private reader.

            :param metadata: The CollectionMetadata object with information about all tables referenced in this query
            :param reader: The data reader to wrap, such as a SqlServerReader, PandasReader, or SparkReader
                The PrivateReader intercepts queries to the underlying reader and ensures differential privacy.
            :param epsilon_per_column: The privacy budget to spend for each column in the query
            :param delta: The delta privacy parameter
            :param interval_widths: If supplied, returns confidence intervals of the specified width, e.g. [0.95, 0.75]
            :param options: A PrivateReaderOptions with flags that change the behavior of the privacy
                engine.
        """
        # check for old calling convention
        if isinstance(metadata, Reader):
            warnings.warn(
                "[reader] API has changed to pass (reader, metadata).  Please update code to pass reader first and metadata second.  This will be a breaking change in future versions.",
                Warning,
            )
            tmp = reader
            reader = metadata
            metadata = tmp

        if isinstance(reader, Reader):
            self.reader = reader
        else:
            raise ValueError("Parameter reader must be of type Reader")

        # using string here, because we don't want to import .metadata due to circular reference
        if "metadata.collection.CollectionMetadata" in str(type(metadata)):
            self.metadata = metadata
        else:
            raise ValueError(
                "Parameter metadata must be of type CollectionMetadata. Got {0}",
                str(type(metadata)),
            )

        self.rewriter = Rewriter(metadata)
        self.epsilon_per_column = epsilon_per_column
        if epsilon is not None:
            message = (
                "epsilon named parameter was replaced with "
                "epsilon_per_column to be more descriptive."
            )
            if epsilon != epsilon_per_column:
                raise Exception(message)
            else:
                module_logger.warning(message)
        if options is not None:
            raise ValueError("Options has been deprecated.  Use metadata")
        self._options = PrivateReaderOptions()

        self.delta = delta
        self.interval_widths = interval_widths
        self._cached_exact = None
        self._cached_ast = None
        self.refresh_options()

    @property
    def epsilon(self):
        module_logger.warning(
            "Epsilon property will be replaced with "
            "the more descriptive epsilon_per_column property."
        )
        return self.epsilon_per_column

    @property
    def engine(self):
        return self.reader.engine

    def refresh_options(self):
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

    @staticmethod
    def get_budget_multiplier(schema, reader, query):
        return len(PrivateReader(reader, schema, 1).get_privacy_cost(query))

    def parse_query_string(self, query_string):
        queries = QueryParser(self.metadata).queries(query_string)
        if len(queries) > 1:
            raise ValueError("Too many queries provided.  We can only execute one query at a time.")
        elif len(queries) == 0:
            return []
        return queries[0]

    def rewrite(self, query_string):
        query = self.parse_query_string(query_string)
        return self.rewrite_ast(query)

    def rewrite_ast(self, query):
        query_max_contrib = query.max_ids
        if self._options.max_contrib is None or self._options.max_contrib > query_max_contrib:
            self._options.max_contrib = query_max_contrib

        self.refresh_options()
        query = self.rewriter.query(query)
        subquery = query.source.relations[0].primary.query
        return (subquery, query)

    def get_privacy_cost(self, query_string):
        self.refresh_options()
        subquery, query = self.rewrite(query_string)
        return subquery.numeric_symbols()

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

    def execute(self, query_string):
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
        return self._execute_ast(query)

    def _execute_ast(self, query, cache_exact=False):
        if isinstance(query, str):
            raise ValueError("Please pass AST to _execute_ast.")

        subquery, query = self.rewrite_ast(query)
        max_contrib = self._options.max_contrib if self._options.max_contrib is not None else 1
        thresh_scale = math.sqrt(max_contrib) * (
            (
                math.sqrt(math.log(1 / self.delta))
                + math.sqrt(math.log(1 / self.delta) + self.epsilon_per_column)
            )
            / (math.sqrt(2) * self.epsilon_per_column)
        )
        self.tau = 1 + thresh_scale * math.sqrt(
            2 * math.log(max_contrib / math.sqrt(2 * math.pi * self.delta))
        )

        syms = subquery.all_symbols()
        source_col_names = [s[0] for s in syms]

        # list of sensitivities in column order
        sens = [s[1].sensitivity() for s in syms]

        # tell which are counts, in column order
        is_count = [s[1].is_count for s in syms]

        # set sensitivity to None if the column is a grouping key
        if subquery.agg is not None:
            group_keys = [
                ge.expression.name if hasattr(ge.expression, "name") else None
                for ge in subquery.agg.groupingExpressions
            ]
        else:
            group_keys = []
        is_group_key = [colname in group_keys for colname in [s[0] for s in syms]]
        for idx in range(len(sens)):
            if is_group_key[idx]:
                sens[idx] = None

        if any([s is np.inf for s in sens]):
            raise ValueError(
                "Query is attempting to query an unbounded column that isn't part of the grouping key"
            )

        kc_pos = None
        for idx in range(len(syms)):
            sname, sym = syms[idx]
            if sname == "keycount":
                kc_pos = idx

        # make a list of mechanisms in column order
        mechs = [
            Gaussian(self.epsilon_per_column, self.delta, s, max_contrib) if s is not None else None
            for s in sens
        ]

        # execute the subquery against the backend and load in tuples
        if cache_exact:
            # we only execute the exact query once
            if self._cached_exact is not None:
                if subquery == self._cached_ast:
                    db_rs = self._cached_exact
                else:
                    raise ValueError(
                        "Cannot run different query against cached result.  "
                        "Make a new PrivateReader or else clear the cache with cache = False"
                    )
            else:
                db_rs = self._get_reader(subquery)._execute_ast(subquery)
                self._cached_exact = list(db_rs)
                self._cached_ast = subquery
        else:
            self.cached_exact = None
            self.cached_ast = None
            db_rs = self._get_reader(subquery)._execute_ast(subquery)

        clamp_counts = self._options.clamp_counts

        def process_row(row_in):
            # pull out tuple values
            row = [v for v in row_in]
            # set null to 0 before adding noise
            for idx in range(len(row)):
                if sens[idx] is not None and row[idx] is None:
                    row[idx] = 0.0
            # call all mechanisms to add noise
            out_row = [
                noise.release([v]).values[0] if noise is not None else v
                for noise, v in zip(mechs, row)
            ]
            # clamp counts to be non-negative
            if clamp_counts:
                for idx in range(len(row)):
                    if is_count[idx] and out_row[idx] < 0:
                        out_row[idx] = 0
            return out_row

        if hasattr(db_rs, "rdd"):
            # it's a dataframe
            out = db_rs.rdd.map(process_row)
        elif hasattr(db_rs, "map"):
            # it's an RDD
            out = db_rs.map(process_row)
        else:
            out = map(process_row, db_rs[1:])

        if subquery.agg is not None and self._options.censor_dims:
            if kc_pos == None:
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

        def process_out_row(row):
            bindings = dict((name.lower(), val) for name, val in zip(source_col_names, row))
            row = [c.expression.evaluate(bindings) for c in query.select.namedExpressions]
            return [convert(val, type) for val, type in zip(row, out_types)]

        if hasattr(out, "map"):
            # it's an RDD
            out = out.map(process_out_row)
        else:
            out = map(process_out_row, out)

        def filter_aggregate(row, condition):
            bindings = dict((name.lower(), val) for name, val in zip(out_col_names, row))
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
                return tuple(
                    [
                        row[idx]
                        if not desc
                        else not row[idx]
                        if out_types[idx] == "boolean"
                        else -row[idx]
                        for desc, idx in sort_fields
                    ]
                )

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
                if hasattr(db_rs, "rdd"):
                    # it's a dataframe
                    out = db_rs.limit(limit_rows)
                elif hasattr(db_rs, "map"):
                    # it's an RDD
                    out = db_rs.limit(limit_rows)
                else:
                    out = itertools.islice(out, limit_rows)

        # output it
        if hasattr(out, "toDF"):
            # Pipeline RDD
            return out.toDF(out_col_names)
        elif hasattr(out, "map"):
            # Bare RDD
            return out
        else:
            out_rows = [out_col_names] + list(out)
            return out_rows

    def _execute_ast_df(self, query, cache_exact=False):
        return self._to_df(self._execute_ast(query, cache_exact))


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
