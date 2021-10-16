import os
import subprocess
import pandas as pd
from itertools import islice
from sqlalchemy.sql.sqltypes import Float

from tqdm import tqdm

root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

from sqlalchemy import Table, Column, Integer, Float, Boolean, MetaData

def create_pums(engine):
    pums_csv_path = os.path.join(root_url, "datasets", "PUMS.csv")
    with engine.begin():
        metadata_obj = MetaData()
        pums = Table('pums', metadata_obj,
            Column('age', Integer),
            Column('sex', Integer),
            Column('educ', Integer),
            Column('race', Integer),
            Column('income', Integer),
            Column('married', Integer)
        )
        if engine.dialect.has_table(engine.connect(), 'pums'):
            pums.drop(engine)
        metadata_obj.create_all(engine)
    pums_df = pd.read_csv(pums_csv_path)
    with engine.begin() as conn:
        for _, row in pums_df.iterrows():
            ins = pums.insert().values(
                sex = int(row['sex']),
                age = int(row['age']),
                educ = int(row['educ']),
                race = int(row['race']),
                income = int(row['income']),
                married = int(row['married'])
            )
            conn.execute(ins)
 
def create_pums_pid(engine):
    pums_pid_csv_path = os.path.join(root_url, "datasets", "PUMS_pid.csv")
    with engine.begin():
        metadata_obj = MetaData()
        pums = Table('pums', metadata_obj,
            Column('pid', Integer, primary_key=True),
            Column('age', Integer),
            Column('sex', Integer),
            Column('educ', Integer),
            Column('race', Integer),
            Column('income', Integer),
            Column('married', Integer)
        )
        if engine.dialect.has_table(engine.connect(), 'pums'):
            pums.drop(engine)
        metadata_obj.create_all(engine)
    pums_df = pd.read_csv(pums_pid_csv_path)
    with engine.begin() as conn:
        for _, row in pums_df.iterrows():
            ins = pums.insert().values(
                pid = int(row['pid']),
                sex = int(row['sex']),
                age = int(row['age']),
                educ = int(row['educ']),
                race = int(row['race']),
                income = int(row['income']),
                married = int(row['married'])
            )
            conn.execute(ins)

def create_pums_dup(engine):
    pums_dup_csv_path = os.path.join(root_url, "datasets", "PUMS_dup.csv")
    with engine.begin():
        metadata_obj = MetaData()
        pums = Table('pums', metadata_obj,
            Column('pid', Integer),
            Column('age', Integer),
            Column('sex', Integer),
            Column('educ', Integer),
            Column('race', Integer),
            Column('income', Integer),
            Column('married', Integer)
        )
        if engine.dialect.has_table(engine.connect(), 'pums'):
            pums.drop(engine)
        metadata_obj.create_all(engine)
    pums_df = pd.read_csv(pums_dup_csv_path)
    with engine.begin() as conn:
        for _, row in pums_df.iterrows():
            ins = pums.insert().values(
                pid = row['pid'],
                sex = row['sex'],
                age = row['age'],
                educ = row['educ'],
                race = row['race'],
                income = row['income'],
                married = row['married']
            )
            conn.execute(ins)

def create_pums_null(engine):
    pums_null_csv_path = os.path.join(root_url, "datasets", "PUMS_null.csv")
    with engine.begin():
        metadata_obj = MetaData()
        pums = Table('pums', metadata_obj,
            Column('pid', Integer, nullable=True),
            Column('age', Integer, nullable=True),
            Column('sex', Integer, nullable=True),
            Column('educ', Integer, nullable=True),
            Column('race', Integer, nullable=True),
            Column('income', Integer, nullable=True),
            Column('married', Integer, nullable=True)
        )
        if engine.dialect.has_table(engine.connect(), 'pums'):
            pums.drop(engine)
        metadata_obj.create_all(engine)
    pums_df = pd.read_csv(pums_null_csv_path)
    with engine.begin() as conn:
        for _, row in pums_df.iterrows():
            ins = pums.insert().values(
                pid = row['pid'] if not pd.isnull(row['pid']) else None,
                sex = row['sex'] if not pd.isnull(row['sex']) else None,
                age = row['age'] if not pd.isnull(row['age']) else None,
                educ = row['educ'] if not pd.isnull(row['educ']) else None,
                race = row['race'] if not pd.isnull(row['race']) else None,
                income = row['income'] if not pd.isnull(row['income']) else None,
                married = row['married'] if not pd.isnull(row['married']) else None
            )
            conn.execute(ins)

def create_pums_large(engine):
    pums_large_csv_path = os.path.join(root_url, "datasets", "PUMS_large.csv")
    with engine.begin():
        metadata_obj = MetaData()
        pums_large = Table('pums_large', metadata_obj,
            Column('PersonID', Integer, primary_key=True),
            Column('state', Integer),
            Column('puma', Integer), 
            Column('sex', Integer),
            Column('age', Integer),
            Column('educ', Integer),
            Column('income', Float),
            Column('latino', Boolean),
            Column('black', Boolean),
            Column('asian', Boolean),
            Column('married', Boolean)
        )
        if engine.dialect.has_table(engine.connect(), 'pums_large'):
            pums_large.drop(engine)
        metadata_obj.create_all(engine)
    with open(pums_large_csv_path) as pums:
        next(pums)
        batch_size = 500
        pbar = tqdm(total=1_200_000)
        while True:
            rows = list(islice(pums, batch_size))
            if not rows:
                break
            else:
                with engine.begin() as conn:
                    row_vals = []
                    for row in rows:
                        personid, state, puma, sex, age, educ, income, latino, black, asian, married = [s.strip() for s in row.split(',')]
                        d = dict(
                            PersonID = int(personid.replace('"', '').replace("'", '')),
                            state = int(state),
                            puma = int(puma),
                            sex = int(sex),
                            age = int(age),
                            educ = int(educ),
                            income = float(income),
                            latino = True if latino else False,
                            black = True if black else False,
                            asian = True if asian else False,
                            married = True if married else False
                        )
                        row_vals.append(d)
                    conn.execute(pums_large.insert(), row_vals)
                pbar.update(batch_size)

