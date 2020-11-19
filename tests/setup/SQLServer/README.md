# SQL Server Setup

Run `python install.py` to install the SQL Server tables.  The cleaning script removes the header, drops the empty row at the end, and removes double quotes, then creates all of the database objects and inserts the rows.  The script expects that you have cloned the `opendp-test-datasets` repository as a sibling to the `smartnoise-sdk` folder.  If you have cloned the test datasets to a different folder, edit the script to point to the right location.

The script uses the password set in `SA_PASSWORD` environment variable.  You can edit the script to update the connection information if you want to override port or set password directly in the script.

```
> python install.cmd
Changed database context to 'PUMS'.

[Loading PUMS.PUMS |################################| 1001/1001.
[Loading PUMS.PUMS_large |################################| 1223993/1223993.
```