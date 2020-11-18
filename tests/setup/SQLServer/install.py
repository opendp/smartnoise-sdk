import pyodbc
import os
import sys
from progress.bar import Bar

datasets= "../../../../dp-test-datasets/data"
server = 'localhost' 
database = 'PUMS' 
username = 'sa' 
password = os.environ.get('SA_PASSWORD')
max_errors = 10

if password is None:
    raise ValueError("Please set an environment variable SA_PASSWORD, or edit this script to set the password.")

with os.popen("sqlcmd -S localhost -U sa -P {0} -i CREATE.sql".format(password)) as out:
    for line in out:
        print(line)

cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()

def load_data(tablename, colnames, filename, max_errors=10, types=None, skip_header=True):
    names = colnames
    n_cols = len(names.split(","))
    with open(filename) as f:
        for nlines, l in enumerate(f):
            pass
    total_lines = nlines + 1

    with open(filename, "r") as f:
        lines_read = 0
        lines_written = 0
        errors = 0
        with Bar("Loading " + tablename, max=total_lines) as bar:
            for line in f:
                lines_read += 1
                bar.next()
                if lines_read % 100 == 0:
                    sys.stderr.write('.')
                    sys.stderr.flush()
                if lines_read == 1 and skip_header:
                    continue
                cols = line.strip().replace('"','').split(',')
                if len(cols) == n_cols:
                    vals = ", ".join(cols)
                    insert = "INSERT INTO {0} ({1}) VALUES ({2})".format(tablename, names, vals)
                    try:
                        cursor.execute(insert)
                        cursor.commit()
                        lines_written += 1
                    except pyodbc.Error as ex:
                        print("\nError inserting from line {0}: {1}".format(lines_read, str(ex)))
                        errors += 1
                        if errors > max_errors:
                            raise RuntimeError("Maximum errors exceeded")

# Now load the databases
pums_path = os.path.join(os.getcwd(), datasets, "PUMS_California_Demographics_1000/data.csv")
colnames = "age, sex, educ, race, income, married"
load_data("PUMS.PUMS", colnames, pums_path)

pums_path = os.path.join(os.getcwd(), datasets, "PUMS_California_Demographics/data.csv")
colnames = "PersonID, state, puma, sex, age, educ, income, latino, black, asian, married"
load_data("PUMS.PUMS_large", colnames, pums_path)

