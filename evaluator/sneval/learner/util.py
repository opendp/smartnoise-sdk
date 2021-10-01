import numpy as np
import pandas as pd
import copy
import csv
import os
from opendp.smartnoise.metadata.collection import *
from opendp.smartnoise.evaluation.learner._generate import Grammar


def create_simulated_dataset(dataset_size, file_name):
    """
    Returns a simulated dataset of configurable size and following
    geometric distribution. Adds a couple of dimension columns for
    algorithm related to GROUP BY queries.
    """
    np.random.seed(1)
    userids = list(range(1, dataset_size + 1))
    userids = ["A" + str(user) for user in userids]
    segment = ["A", "B", "C"]
    role = ["R1", "R2"]
    roles = np.random.choice(role, size=dataset_size, p=[0.7, 0.3]).tolist()
    segments = np.random.choice(segment, size=dataset_size, p=[0.5, 0.3, 0.2]).tolist()
    usage = np.random.geometric(p=0.5, size=dataset_size).tolist()
    df = pd.DataFrame(
        list(zip(userids, segments, roles, usage)), columns=["UserId", "Segment", "Role", "Usage"]
    )

    # Storing the data as a CSV
    metadata = Table(
        file_name,
        file_name,
        [
            String("UserId", dataset_size, True),
            String("Segment", 3, False),
            String("Role", 2, False),
            Int("Usage", 0, 25),
        ],
        dataset_size,
    )

    return df, metadata


def generate_neighbors(df, metadata, flag="bandit"):
    """
    Generate dataframes that differ by a single record that is randomly chosen
    Returns the neighboring datasets and their corresponding metadata
    """
    d1 = df
    drop_idx = np.random.choice(df.index, 1, replace=False)
    d2 = df.drop(drop_idx)
    d1_table = metadata
    d2_table = copy.copy(d1_table)
    d1_table.schema, d2_table.schema = "dataset", "dataset"
    d1_table.name, d2_table.name = "dataset", "dataset"
    d2_table.rowcount = d1_table.rowcount - 1
    d1_metadata, d2_metadata = (
        CollectionMetadata([d1_table], "csv"),
        CollectionMetadata([d2_table], "csv"),
    )

    return d1, d2, d1_metadata, d2_metadata


def generate_query(numofquery, select_path):
    # generate query pool
    print(select_path)
    with open(select_path, "r") as cfg:
        rules = cfg.readlines()
        grammar = Grammar(numofquery)
        grammar.load(rules)

    querypool = []
    for i in range(numofquery):
        querypool.append(str(grammar.generate("statement")))
    return querypool


def write_to_csv(filename, data, flag):
    with open(filename, "w", newline="") as csvfile:
        if flag == "qlearning":
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "original_query",
                    "chosen_action",
                    "new_query",
                    "episode",
                    "dpresult",
                    "reward",
                    "message",
                    "d1",
                    "d2",
                ],
                extrasaction="ignore",
            )
        elif flag == "bandit":
            writer = csv.DictWriter(
                csvfile, fieldnames=["query", "dpresult", "jensen_shannon_divergence", "error"]
            )
        writer.writeheader()
        for i in data:
            writer.writerow(i)
