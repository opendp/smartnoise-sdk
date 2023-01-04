import os
import subprocess
import sys
from urllib.parse import quote_plus

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

engine.execute("CREATE DATABASE IF NOT EXISTS PUMS")
engine.execute("CREATE DATABASE IF NOT EXISTS PUMS_pid")
engine.execute("CREATE DATABASE IF NOT EXISTS PUMS_dup")
engine.execute("CREATE DATABASE IF NOT EXISTS PUMS_null")
engine.execute("CREATE DATABASE IF NOT EXISTS PUMS_large")

dburl = url + "/PUMS"
engine = create_engine(dburl)
create_pums(engine)
with engine.begin() as conn:
    count = list(conn.execute('SELECT COUNT(*) FROM pums'))
    print(f"PUMS has {count[0][0]} rows")

dburl = url + "/PUMS_pid"
engine = create_engine(dburl)
create_pums_pid(engine)
with engine.begin() as conn:
    count = list(conn.execute('SELECT COUNT(*) FROM pums'))
    print(f"PUMS_pid has {count[0][0]} rows")

dburl = url + "/PUMS_dup"
engine = create_engine(dburl)
create_pums_dup(engine)
with engine.begin() as conn:
    count = list(conn.execute('SELECT COUNT(*) FROM pums'))
    print(f"PUMS_pid has {count[0][0]} rows")

dburl = url + "/PUMS_null"
engine = create_engine(dburl)
create_pums_null(engine)
with engine.begin() as conn:
    count = list(conn.execute('SELECT COUNT(*) FROM pums'))
    print(f"PUMS_null has {count[0][0]} rows")

dburl = url + "/PUMS_large"
engine = create_engine(dburl)
create_pums_large(engine)
with engine.begin() as conn:
    count = list(conn.execute('SELECT COUNT(*) FROM pums_large'))
    print(f"PUMS_large has {count[0][0]} rows")
