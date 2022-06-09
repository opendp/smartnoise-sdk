Our database tests in CI will test against pandas, postgres, sqlserver, and spark.  Developers are encouraged to test against local installed databases during unit testing, since this is much faster than CI, and can debug against multiple engines.

If no additional databases are available, the tests will use test tables via pandas, and setting `export TEST_SPARK=1` will enable tests against tables using pyspark.  To include other SQL engines, edit `~/.smartnoise/connections-unit.yaml`.  The passwords can be saved in the local keyring, or in environment variables.  To set passwords in the keyring, use `python tests/sdk/engines/engine-creds.py`.  To set passwords in environment variables, use the engine name used by SmartNoise.  For example, the test harness will use `POSTGRES_PASSWORD` if set.

Test harness will enumerate connections listed in config, and only run unit tests for engines where a connection is available.

For more on using test databases from unit tests, [see the test README](../../README.md).