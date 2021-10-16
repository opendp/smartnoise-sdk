sqlcmd -S "(localdb)\MSSQLLocalDB" -i pums.sql
python clean_text.py
sqlcmd -S "(localdb)\MSSQLLocalDB" -i install.sql
del PUMS*.csv