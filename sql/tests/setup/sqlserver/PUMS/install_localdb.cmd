sqlcmd -l 90 -S "(localdb)\MSSQLLocalDB" -i pums.sql
python clean_text.py
sqlcmd -l 90 -S "(localdb)\MSSQLLocalDB" -i install.sql
del PUMS*.csv