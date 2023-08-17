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

if not os.path.exists(pums_path):
    raise FileNotFoundError("PUMS.csv not found. Make sure you have run `git lfs pull`.")
if not os.path.exists(pums_id_path):
    raise FileNotFoundError("PUMS_null.csv not found. Make sure you have run `git lfs pull`.")
if not os.path.exists(pums_large_path):
    raise FileNotFoundError("PUMS_large.csv not found. Make sure you have run `git lfs pull`.")


def read_pums_large_csv(spark_session, pums_large_path):
    df = spark.read.csv(pums_large_path, header=True)
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

@pytest.fixture(scope="module")
def pums_df(spark_session):
    df = spark_session.read.csv(pums_path, header=True)
    df = df.withColumn("income", df["income"].cast("int"))
    df = df.withColumn("age", df["age"].cast("int"))
    df = df.withColumn("educ", df["educ"].cast("int"))
    df = df.withColumn("married", df["married"].cast("boolean"))
    return df

@pytest.fixture(scope="module")
def pums_id_df(spark_session):
    df = spark_session.read.csv(pums_id_path, header=True)
    df = df.withColumn("income", df["income"].cast("int"))
    df = df.withColumn("age", df["age"].cast("int"))
    df = df.withColumn("educ", df["educ"].cast("int"))
    df = df.withColumn("married", df["married"].cast("boolean"))
    return df

@pytest.fixture(scope="module")
def pums_large_df(spark_session):
    df = read_pums_large_csv(spark_session, pums_large_path)
    return df

@pytest.fixture(scope="module")
def pums_agg_df(spark_session):
    df = spark_session.read.parquet(pums_agg_path)
    return df

@pytest.fixture(scope="module")
def pums_dataset(pums_df):
    return Dataset(
        pums_df,
        categorical_columns=["age", "sex", "educ", "race", "married"],
        measure_columns=["income"]
        )

@pytest.fixture(scope="module")
def pums_id_dataset(pums_id_df):
    return Dataset(
        pums_id_df,
        categorical_columns=["age", "sex", "educ", "race", "married"],
        measure_columns=["income"],
        id_column="pid"
        )

@pytest.fixture(scope="module")
def pums_agg_dataset(pums_agg_df):
    return Dataset(
        pums_agg_df,
        categorical_columns=["sex","age","educ","latino","black","asian","married"],
        sum_columns=["income"],
        count_column="count"
        )
                   