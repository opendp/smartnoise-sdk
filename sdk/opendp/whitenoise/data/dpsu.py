import pandas as pd
import math
import random
import operator
from collections import defaultdict

from opendp.whitenoise.metadata import CollectionMetadata
from opendp.whitenoise.sql import PandasReader
from opendp.whitenoise.sql.parse import QueryParser
from opendp.whitenoise.mechanisms.rand import laplace

def preprocess_df_from_query(meta_path, csv_path, query, table_name):
	"""
	Returns a dataframe with user_id | tuple based on query grouping keys.
	"""
	schema = CollectionMetadata.from_file(meta_path)
	df = pd.read_csv(csv_path)

	reader = PandasReader(schema, df)
	queries = QueryParser(schema).queries(query)
	query_ast = queries[0]
	
	group_cols = [ge.expression.name for ge in query_ast.agg.groupingExpressions]
	kc = schema[table_name].key_cols()[0].name

	preprocessed_df = pd.DataFrame()
	preprocessed_df[kc] = df[kc]
	preprocessed_df["group_cols"] = tuple(df[group_cols].values.tolist())

	return preprocessed_df

def reservoir_sample(iterable, max_contrib):
	reservoir = []
	for i, item in enumerate(iterable):
		if i < max_contrib:
			reservoir.append(item)
		else:
			m = random.randint(0, i)
			if m < max_contrib:
				reservoir[m] = item
	
	return reservoir

def policy_laplace(df, max_contrib, eps, delta):
	"""
	Differentially Private Set Union

	Given a database of n users, each with a subset of items,
	(eps, delta)-differentially private algorithm that outputs the largest possible set of the
	the union of these items.

	Parameters
	----------
	max_contrib: maximum number of items a user can contribute
	epsilon/delta: privacy parameters
	"""
	alpha = 3.0
	lambd = 1 / eps
	rho = [1/i + lambd * math.log(1 / (2 * (1 - (1 - delta)**(1 / i)))) for i in range(1, max_contrib+1)]
	rho = max(rho)
	gamma = rho + alpha * lambd

	histogram = defaultdict(float)

	key_col = df.columns[0]
	df["hash"] = df[key_col].apply(lambda x: hash(str(x)))
	df = df.sort_values("hash")

	for idx, group in df.groupby(key_col):
		items = group["group_cols"]

		if len(items) > max_contrib:
			items = reservoir_sample(items, max_contrib)

		cost_dict = {}
		for u in items:
			if histogram[u] < gamma:
				cost_dict[u] = gamma - histogram[u]

		budget = 1
		k = len(cost_dict)

		cost_dict = {k: v for k, v in sorted(cost_dict.items(), key=lambda item: item[1])}
		cost_keys = list(cost_dict.keys())

		for i, w in enumerate(cost_keys):
			cost = cost_dict[w] * k
			if cost <= budget:
				for j in range(i, k):
					item_bin = cost_keys[j]
					histogram[item_bin] += cost_dict[w]
					cost_dict[item_bin] -= cost_dict[w]
				budget -= cost
				k -= 1
			else:
				for j in range(i, k):
					item_bin = cost_keys[j]
					histogram[item_bin] += budget / k
				break

	items = []
	for u in histogram.keys():
		histogram[u] += laplace(0, lambd, 1)[0]
		if histogram[u] > rho:
			items.append(u)

	df= df[df["group_cols"].isin(items)]
	return df

def dpsu_df(df1, df2):
	"""

	"""
	dpsu_df = pd.merge(df1, df2, on=df2.columns[0])
	dpsu_df.drop(["group_cols", "hash"], axis=1, inplace=True)

	return dpsu_df