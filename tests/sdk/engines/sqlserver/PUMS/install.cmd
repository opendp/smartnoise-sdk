REM set the password with SET SAPASSWORD=...
sqlcmd -U sa  -P %SAPASSWORD% -i pums.sql
python set_import_path.py
sqlcmd -U sa  -P %SAPASSWORD% -i install.sql
del PUMS*.csv