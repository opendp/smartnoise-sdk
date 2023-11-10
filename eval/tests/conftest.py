import os
import subprocess
import pyspark
import pytest
from pyspark.sql import functions as F
from sneval import Dataset

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

pums_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))
pums_id_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_null.csv"))
pums_large_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_large.csv"))
pums_agg_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_large.parquet"))
test_all_plus_last_6k_path = os.path.join(git_root_dir, os.path.join("datasets", "TEST_all_plus_last_6k.csv"))
test_all_minus_last_12k_path = os.path.join(git_root_dir, os.path.join("datasets", "TEST_all_minus_last_12k.csv"))

if not os.path.exists(pums_path):
    raise FileNotFoundError("PUMS.csv not found. Make sure you have run `git lfs pull`.")
if not os.path.exists(pums_id_path):
    raise FileNotFoundError("PUMS_null.csv not found. Make sure you have run `git lfs pull`.")
if not os.path.exists(pums_large_path):
    raise FileNotFoundError("PUMS_large.csv not found. Make sure you have run `git lfs pull`.")
if not os.path.exists(test_all_plus_last_6k_path):
    raise FileNotFoundError("TEST_all_plus_last_6k.csv not found. Make sure you have run `git lfs pull`.")
if not os.path.exists(test_all_minus_last_12k_path):
    raise FileNotFoundError("TEST_all_minus_last_12k.csv not found. Make sure you have run `git lfs pull`.")


def read_pums_large_csv(spark_session, pums_large_path):
    df = spark_session.read.csv(pums_large_path, header=True)
    df = df.select(["sex","age","educ","married","latino","black","asian","income"])
    df = df.withColumn("income", df["income"].cast("int"))
    df = df.withColumn("age", df["age"].cast("int"))
    df = df.withColumn("educ", df["educ"].cast("int"))
    df = df.withColumn("married", df["married"].cast("boolean"))
    df = df.withColumn("latino", df["latino"].cast("boolean"))
    df = df.withColumn("black", df["black"].cast("boolean"))
    df = df.withColumn("asian", df["asian"].cast("boolean"))
    return df

if not os.path.exists(pums_agg_path):
    # create the parquet file
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = read_pums_large_csv(spark, pums_large_path)
    categories = [c for c in df.columns if c not in ["income"]]
    df = df.groupBy(categories).agg(F.count('*').alias('count'),F.sum('income').alias('income'))
    df.write.parquet(pums_agg_path)
    spark.stop()

@pytest.fixture(scope="module")
def spark_session():
    spark = pyspark.sql.SparkSession.builder \
        .appName("PySpark Test") \
        .getOrCreate()
    yield spark
    spark.stop()

# PUMS.csv
# --------
def pums_df_(spark_session):
    df = spark_session.read.csv(pums_path, header=True)
    df = df.withColumn("income", df["income"].cast("int"))
    df = df.withColumn("age", df["age"].cast("int"))
    df = df.withColumn("educ", df["educ"].cast("int"))
    df = df.withColumn("married", df["married"].cast("boolean"))
    return df

@pytest.fixture(scope="module")
def pums_df(spark_session):
    df = pums_df_(spark_session)
    return df

# PUMS_id.csv
# -----------
def pums_id_df_(spark_session):
    df = spark_session.read.csv(pums_id_path, header=True)
    df = df.withColumn("income", df["income"].cast("int"))
    df = df.withColumn("age", df["age"].cast("int"))
    df = df.withColumn("educ", df["educ"].cast("int"))
    df = df.withColumn("married", df["married"].cast("boolean"))
    return df

@pytest.fixture(scope="module")
def pums_id_df(spark_session):
    df = pums_id_df_(spark_session)
    return df

# PUMS_large.csv
# --------------
def pums_large_df_(spark_session):
    df = read_pums_large_csv(spark_session, pums_large_path)
    return df

@pytest.fixture(scope="module")
def pums_large_df(spark_session):
    df = pums_large_df_(spark_session)
    return df

# PUMS_large.parquet
# ------------------
def pums_large_df_(spark_session):
    df = read_pums_large_csv(spark_session, pums_large_path)
    return df

@pytest.fixture(scope="module")
def pums_agg_df(spark_session):
    df = pums_large_df_(spark_session)
    return df

# --------
# DATASETS
# --------

# PUMS.csv
def pums_dataset_(pums_df):
    return Dataset(
        pums_df,
        categorical_columns=["age", "sex", "educ", "race", "married"],
        measure_columns=["income"]
        )

@pytest.fixture(scope="module")
def pums_dataset(pums_df):
    return pums_dataset_(pums_df)

# PUMS_id.csv
def pums_id_dataset_(pums_id_df):
    return Dataset(
        pums_id_df,
        categorical_columns=["age", "sex", "educ", "race", "married"],
        measure_columns=["income"],
        id_column="pid"
        )

@pytest.fixture(scope="module")
def pums_id_dataset(pums_id_df):
    return pums_id_dataset_(pums_id_df)

# PUMS_large.csv
def pums_large_dataset_(pums_large_df):
    return Dataset(
        pums_large_df,
        categorical_columns=["sex","age","educ","latino","black","asian","married"],
        measure_columns=["income"],
        )

@pytest.fixture(scope="module")
def pums_large_dataset(pums_large_df):
    return pums_large_dataset_(pums_large_df)

# PUMS_large.parquet
def pums_agg_dataset_(pums_agg_df):
    return Dataset(
        pums_agg_df,
        categorical_columns=["sex","age","educ","latino","black","asian","married"],
        sum_columns=["income"],
        count_column="count"
        )

@pytest.fixture(scope="module")
def pums_agg_dataset(pums_agg_df):
    return pums_agg_dataset_(pums_agg_df)

@pytest.fixture(scope="module")
def test_all_plus_last_6k_df(spark_session):
    df = spark_session.read.csv(test_all_plus_last_6k_path, header=True)
    df = df.select(["ProductID","CustomerRegion"])
    df = df.withColumn("ProductID", df["ProductID"].cast("int"))
    df = df.withColumn("CustomerRegion", df["CustomerRegion"].cast("int"))
    return df

@pytest.fixture(scope="module")
def test_all_plus_last_6k_dataset(test_all_plus_last_6k_df):
    return Dataset(
        test_all_plus_last_6k_df,
        categorical_columns=["ProductID","CustomerRegion"]
        )

@pytest.fixture(scope="module")
def test_all_minus_last_12k_df(spark_session):
    df = spark_session.read.csv(test_all_minus_last_12k_path, header=True)
    df = df.select(["ProductID","CustomerRegion"])
    df = df.withColumn("ProductID", df["ProductID"].cast("int"))
    df = df.withColumn("CustomerRegion", df["CustomerRegion"].cast("int"))
    return df

@pytest.fixture(scope="module")
def test_all_minus_last_12k_dataset(test_all_minus_last_12k_df):
    return Dataset(
        test_all_minus_last_12k_df,
        categorical_columns=["ProductID","CustomerRegion"]
        )

                   