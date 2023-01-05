import os
import subprocess

root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

from sqlalchemy import create_engine, text

from .create_pums_dbs import *

def make_sqlite():
    pums_db_path = os.path.join(root_url, "datasets", "PUMS.db")
    if not os.path.exists(pums_db_path):
        print('Creating PUMS in SQLite')
        sqlite_file_path = f'sqlite:///{pums_db_path}'
        engine = create_engine(sqlite_file_path, echo=False)
        create_pums(engine)
        with engine.begin() as conn:
            res = conn.execute(text('SELECT COUNT(*) FROM pums'))
            print(list(res))

    pums_pid_db_path = os.path.join(root_url, "datasets", "PUMS_pid.db")
    if not os.path.exists(pums_pid_db_path):
        print('Creating PUMS_pid in SQLite')
        sqlite_file_path = f'sqlite:///{pums_pid_db_path}'
        engine = create_engine(sqlite_file_path, echo=False)
        create_pums_pid(engine)
        with engine.begin() as conn:
            res = conn.execute(text('SELECT COUNT(*) FROM pums'))
            print(list(res))

    pums_dup_db_path = os.path.join(root_url, "datasets", "PUMS_dup.db")
    if not os.path.exists(pums_dup_db_path):
        print('Creating PUMS_dup in SQLite')
        sqlite_file_path = f'sqlite:///{pums_dup_db_path}'
        engine = create_engine(sqlite_file_path, echo=False)
        create_pums_dup(engine)
        with engine.begin() as conn:
            res = conn.execute(text('SELECT COUNT(*) FROM pums'))
            print(list(res))

    pums_null_db_path = os.path.join(root_url, "datasets", "PUMS_null.db")
    if not os.path.exists(pums_null_db_path):
        print('Creating PUMS_null in SQLite')
        sqlite_file_path = f'sqlite:///{pums_null_db_path}'
        engine = create_engine(sqlite_file_path, echo=False)
        create_pums_null(engine)
        with engine.begin() as conn:
            res = conn.execute(text('SELECT COUNT(*) FROM pums'))
            print(list(res))
