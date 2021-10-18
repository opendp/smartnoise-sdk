import pandas as pd
import math
import random
from collections import defaultdict

from snsql.sql.parse import QueryParser
from snsql._ast.ast import Table
from ._mechanisms.rand import laplace

sys_rand = random.SystemRandom()


def preprocess_df_from_query(schema, df, query_string):
    """
    Returns a dataframe with user_id | tuple based on query grouping keys.
    """
    qp = QueryParser(schema)
    q = qp.query(query_string)
    queries = qp.queries(query_string)
    query_ast = queries[0]

    group_cols = [ge.expression.name for ge in query_ast.agg.groupingExpressions]
    table_name = q.source.find_node(Table).name
    key_col = schema[table_name].key_cols()[0].name

    preprocessed_df = pd.DataFrame()
    preprocessed_df[key_col] = df[key_col]
    preprocessed_df["group_cols"] = tuple(df[group_cols].values.tolist())

    return preprocessed_df


def policy_laplace(df, epsilon, delta, max_contrib):
    """
    Differentially Private Set Union: https://arxiv.org/abs/2002.09745

    Given a database of n users, each with a subset of items,
    (epsilon, delta)-differentially private algorithm that outputs the largest possible set of the
    the union of these items.

    df: pandas df with user_id | item where item is a tuple
    max_contrib: maximum number of items a user can contribute
    epsilon/delta: privacy parameters
    """
    alpha = 3.0
    lambd = 1 / epsilon
    rho = [
        1 / i + lambd * math.log(1 / (2 * (1 - (1 - delta) ** (1 / i))))
        for i in range(1, max_contrib + 1)
    ]
    rho = max(rho)
    gamma = rho + alpha * lambd

    histogram = defaultdict(float)

    key_col = df.columns[0]
    df["hash"] = df[key_col].apply(lambda x: hash(str(x)))
    df = df.sort_values("hash")

    for idx, group in df.groupby(key_col):
        items = list(group["group_cols"])

        if len(items) > max_contrib:
            items = sys_rand.sample(items, max_contrib)

        cost_dict = {}
        for item in items:
            if histogram[item] < gamma:
                cost_dict[item] = gamma - histogram[item]

        budget = 1
        k = len(cost_dict)

        sorted_items = [k for k, v in sorted(cost_dict.items(), key=lambda item: item[1])]

        for idx, curr_item in enumerate(sorted_items):
            cost = (
                cost_dict[curr_item] * k
            )  # cost of increasing weights of remaining k items by cost_dict[curr_item]
            if cost <= budget:
                # update weights of remaining k items with cost_dict[curr_item]
                for j in range(idx, k):
                    remaining_item = sorted_items[j]
                    histogram[remaining_item] += cost_dict[curr_item]
                    cost_dict[remaining_item] -= cost_dict[curr_item]

                budget -= cost
                k -= 1
            else:
                # update weights of remaining k items with budget / k
                for j in range(idx, k):
                    remaining_item = sorted_items[j]
                    histogram[remaining_item] += budget / k
                break

    items = []
    for item in histogram.keys():
        histogram[item] += laplace(0, lambd, 1)[0]
        if histogram[item] > rho:
            items.append(item)

    df = df[df["group_cols"].isin(items)]
    return df


def run_dpsu(schema, input_df, query, epsilon, delta=math.exp(-10), max_contrib=5):
    preprocessed_df = preprocess_df_from_query(schema, input_df, query)

    dpsu_df = policy_laplace(preprocessed_df, epsilon, delta, max_contrib)

    output_df = pd.merge(input_df, dpsu_df, on=dpsu_df.columns[0])
    output_df.drop(["group_cols", "hash"], axis=1, inplace=True)
    output_df.drop_duplicates(inplace=True)

    return output_df
