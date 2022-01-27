"""Miscellaneous utility functions."""

import subprocess
import os

import numpy as np
import pandas as pd


def display_tables(tables, max_rows=10, datetime_fmt='%Y-%m-%d %H:%M:%S', row=True):
    """Display mutiple tables side by side on a Jupyter Notebook.

    Args:
        tables (dict[str, DataFrame]):
            ``dict`` containing table names and pandas DataFrames.
        max_rows (int):
            Max rows to show per table. Defaults to 10.
        datetime_fmt (str):
            Format with which to display datetime columns.
    """
    # Import here to avoid making IPython a hard dependency
    from IPython.core.display import HTML

    names = []
    data = []
    for name, table in tables.items():
        table = table.copy()
        for column in table.columns:
            column_data = table[column]
            if column_data.dtype.kind == 'M':
                table[column] = column_data.dt.strftime(datetime_fmt)

        names.append('<td style="text-align:left"><b>{}</b></td>'.format(name))
        data.append('<td>{}</td>'.format(table.head(max_rows).to_html(index=False)))

    if row:
        html = '<table><tr>{}</tr><tr>{}</tr></table>'.format(
            ''.join(names),
            ''.join(data),
        )
    else:
        rows = [
            '<tr>{}</tr><tr>{}</tr>'.format(name, table)
            for name, table in zip(names, data)
        ]
        html = '<table>{}</table>'.format(''.join(rows))

    return HTML(html)

def retrieve_PUMS_data_categorical():
    git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

    csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))

    df = pd.read_csv(csv_path)
    sample_size = len(df)
    df_non_continuous = df[['sex','educ','race','married','pid']].copy()
    return df, df_non_continuous, sample_size

def return_PUMS_metadata():
    meta = {
        "tables":{
            "pums":{
                "primary_key":"pid",
                "fields":{
                    'pid':{
                    'type': 'id', 
                    'subtype': 'integer'
                    },
                    "sex":{
                    'type': 'categorical'
                    },
                    "educ":{
                    "type":"categorical"
                    },
                    "race":{
                    "type":"categorical"
                    },
                    "married":{
                    "type":"categorical"
                    }
                }
            }
        }
    }

    return meta