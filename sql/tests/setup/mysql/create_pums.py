import os
import subprocess
import sys
from urllib.parse import quote_plus
from sqlalchemy import text

print("Installing test databases for PUMS\n")

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
setup_path = os.path.abspath(
    os.path.join(
        git_root_dir, 
        "sql", 
        "tests", 
        "setup"
    )
)

sys.path.insert(0, setup_path)

from dataloader.create_pums_dbs import *

from sqlalchemy import create_engine

# change these to match your install
host = 'localhost'
port = '3306'
user = 'root'

password = os.environ.get('MYSQL_PASSWORD')
if not password:
    print("Please make sure password is set in MYSQL_PASSWORD")
    exit()

url = f"mysql+pymysql://{user}:{quote_plus(password)}@{host}:{port}"
engine = create_engine(url)

conn = engine.connect()

print("Creating databases\n")
conn.execute(text("CREATE DATABASE IF NOT EXISTS PUMS"))
conn.execute(text("CREATE DATABASE IF NOT EXISTS PUMS_pid"))
conn.execute(text("CREATE DATABASE IF NOT EXISTS PUMS_dup"))
conn.execute(text("CREATE DATABASE IF NOT EXISTS PUMS_null"))
conn.execute(text("CREATE DATABASE IF NOT EXISTS PUMS_large"))

print("Creating PUMS table\n")
dburl = url + "/PUMS"
engine = create_engine(dburl)
create_pums(engine)
with engine.begin() as conn:
    count = list(conn.execute(text('SELECT COUNT(*) FROM pums')))
    print(f"PUMS has {count[0][0]} rows")

print("Creating PUMS_pid table\n")
dburl = url + "/PUMS_pid"
engine = create_engine(dburl)
create_pums_pid(engine)
with engine.begin() as conn:
    count = list(conn.execute(text('SELECT COUNT(*) FROM pums')))
    print(f"PUMS_pid has {count[0][0]} rows")

print("Creating PUMS_dup table\n")
dburl = url + "/PUMS_dup"
engine = create_engine(dburl)
create_pums_dup(engine)
with engine.begin() as conn:
    count = list(conn.execute(text('SELECT COUNT(*) FROM pums')))
    print(f"PUMS_pid has {count[0][0]} rows")

print("Creating PUMS_null table\n")
dburl = url + "/PUMS_null"
engine = create_engine(dburl)
create_pums_null(engine)
with engine.begin() as conn:
    count = list(conn.execute(text('SELECT COUNT(*) FROM pums')))
    print(f"PUMS_null has {count[0][0]} rows")

print("Creating PUMS_large table\n")
dburl = url + "/PUMS_large"
engine = create_engine(dburl)
create_pums_large(engine)
with engine.begin() as conn:
    count = list(conn.execute(text('SELECT COUNT(*) FROM pums_large')))
    print(f"PUMS_large has {count[0][0]} rows")
