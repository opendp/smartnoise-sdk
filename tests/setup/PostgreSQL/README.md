# PostgreSQL Setup

First, run `clean.sh` to strip off the headers of the CSV files.  The bash file expects that you have cloned the opendp-test-datasets repository as a sibling to the smartnoise-sdk folder.  If you have cloned the test datasets to a different folder, edit the bash script to point to the right location.

After creating the cleaned copies of `PUMS.csv` and `PUMS_large.csv`, edit `CREATE.sql` to supply the full path of each file in the `\copy` commands.

If you will be running your unit tests from an account that doesn't have administrator privileges (recommended), change the line with `GRANT` to grant permissions to the correct database user account.

After you have edited the DDL, you can copy and paste from `CREATE.sql` into a `psql` command prompt, or run directly from a command line using the syntax in `install.sh`.

```
# psql -U postgres -f CREATE.new.sql
Password for user postgres:
GRANT
psql:CREATE.new.sql:3: NOTICE:  drop cascades to 2 other objects
DETAIL:  drop cascades to table pums.pums
drop cascades to table pums.pums_large
DROP SCHEMA
CREATE SCHEMA
COPY 1000
COPY 1223992```