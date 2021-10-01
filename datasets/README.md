# Datasets for Unit Tests

CSV files with associated metadata (.yaml).  Files are downloaded by 
pytest test fixture on first use.

* PUMS: A 1000 row sample from PUMS (US Census Public Use Microdata).  Metadata has row_privacy set.
* PUMS_pid: A 1000 row sample from PUMS.  Has an extra column, `pid`, a primary key that can be used to bound user contribution.
* PUMS_large: A sample of 1.2 million records from PUMS, which includes a primary key (`PersonId`) and slightly different schema
* iris: The standard iris dataset
* reddit: A collection of n-grams from reddit posts

The PUMS datasets are downloaded from the [dp-test-datasets](https://github.com/opendp/dp-test-datasets) repository.

You are encouraged to use these datasets in unit tests where the data can be accessed from a CSV.  Some of these datasets are also loaded automatically into the SQL database engines installed into engine-specific GitHub Actions images.