# Database Setup

By default, the unit tests for SQL functionality run only against the PandasReader, which uses SQLite over a Pandas DataFrame.

To ensure that functionality works across all supported database engines, developers should install and configure a variety of databases.  The unit tests which test SQL functionality will automatically run all tests against all installed database engines.

This folder includes setup scripts which ensure the required databases and tables are available in each installed database engine.

To let the unit tests know that a test database engine is available, set the relevant environment variables:

## PostgreSQL

* POSTGRES_PASSWORD: required.
* POSTGRES_USER: optional. Defaults to `postgres`.
* POSTGRES_PORT: optional.  Defaults to `5432`.
* POSTGRES_HOST: optional.  Defaults to `localhost`.

## SQL Server

* SQLSERVER_PASSWORD: required.
* SQLSERVER_USER: optional. Defaults to `sa`.
* SQLSERVER_PORT: optional.  Defaults to `1433`.
* SQLSERVER_HOST: optional.  Defaults to `localhost`.

