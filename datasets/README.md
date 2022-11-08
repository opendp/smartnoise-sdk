# Datasets for Unit Tests

CSV files with associated metadata (.yaml).

* PUMS: A 1000 row sample from PUMS (US Census Public Use Microdata).  Metadata has row_privacy set.
* PUMS_pid: A 1000 row sample from PUMS.  Has an extra column, `pid`, a primary key that can be used to bound user contribution.
* PUMS_large: A sample of 1.2 million records from PUMS, which includes a primary key (`PersonId`) and slightly different schema
* PUMS_null: Same as PUMS_pid, with values randomly missing.  Useful for testing nullable support.
* iris: The standard iris dataset
* reddit: A collection of n-grams from reddit posts

# Downloading Datasets

The datasets will be automatically downloaded the first time you run `pytest tests` under `sql/`.  To download the test datasets
without running unit tests, you can do the following:

```python
cd sql
pip install -r tests/requirements.txt
python tests/check_databases.py
```

You are encouraged to use these datasets in unit tests where the data can be accessed from a CSV.  Some of these datasets are also loaded automatically into the SQL database engines installed into engine-specific GitHub Actions images.
