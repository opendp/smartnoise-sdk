# This file contains methods to set and scan a search space of datasets that we shall be using for DP evaluation.
# We need to search databases, neighboring pairs and queries for running the DP predicate test.
# We shall start a small 3-row databases to create our 12 neighboring pairs per database
# Then we shall use halton sequence to generate a set of such 3 row databases randomly in a 3-D log-space
import numpy as np
import pandas as pd
import os
import copy
from statsmodels.tools import sequences
from opendp.whitenoise.metadata.collection import *


class Exploration:
    def __init__(self, dataset_size=3, csv_path=r'../service/datasets/evaluation'):  # TODO changed default to "."
        self.dataset_size = dataset_size
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = csv_path

        self.numerical_col_name = "Usage"
        self.numerical_col_type = "int"
        self.N = dataset_size
        # if we are exploring 3 dimensions, corners shall be [-1000, -1000, -1000] to [1000, 1000, 1000]
        self.corners = np.array([[-1*(10**self.N)]*self.N, [10**self.N]*self.N])
        self.visited = []
        self.neighbor_pair = {}

    # Create a dataset with one numerical column on which we shall evaluate DP queries
    def create_small_dataset(self, sample, file_name = "small"):
        userids = list(range(1, self.dataset_size+1))
        userids = ["A" + str(user) for user in userids]
        usage = list(sample)
        df = pd.DataFrame(list(zip(userids, usage)), columns=['UserId', self.numerical_col_name])
        metadata = Table(file_name, file_name, self.dataset_size, \
        [\
            String("UserId", self.dataset_size, True),\
            Int(self.numerical_col_name, min(usage), max(usage))
        ])
        return df, metadata

    # Generate halton samples in a n-dimensional space. Defaulted to 3 dimensions
    def generate_halton_samples(self, bounds, dims, n_sample=10):
        samples = sequences.halton(dim=dims, n_sample=n_sample, bounds=bounds)
        return samples

    # Given a list of N records in a database, create a powerset of neighboring datasets by traversing the edges of database search graph
    # Perform DFS to traverse the database search graph
    # Convention of file names = <d1/d2>_<list of row indexes in d1>_<row index removed to create d2>
    def generate_powerset(self, d1):
        if(len(d1) == 0):
            return
        else:
            indexes = list(d1.index.values)
            s = [str(i) for i in indexes]
            d1_idx_range = "".join(s)
            for drop_idx in indexes:
                filename = d1_idx_range + "_" + str(drop_idx)
                if(filename not in self.visited):
                    d2 = d1.drop(drop_idx)
                    min_val = min(d1[self.numerical_col_name])
                    max_val = max(d1[self.numerical_col_name])
                    # Avoiding sensitivity to be 0
                    min_val = min_val if max_val > min_val else 0
                    max_val = max_val if max_val > min_val else abs(max_val)

                    d1_table = Table("d1_" + filename, "d1_" + filename, len(d1), \
                    [\
                        String("UserId", len(d1), True),\
                        Int(self.numerical_col_name, min_val, max_val)
                    ])
                    d2_table = copy.copy(d1_table)
                    d2_table.schema, d2_table.name, d2_table.rowcount = "d2_" + filename, "d2_" + filename, d1_table.rowcount - 1
                    d1_metadata, d2_metadata = CollectionMetadata([d1_table], "csv"), CollectionMetadata([d2_table], "csv")

                    self.neighbor_pair[filename] = [d1, d2, d1_metadata, d2_metadata]
                    self.visited.append(filename)
                    self.generate_powerset(d2)
            return

    def test_exploration(self):
        samples = self.generate_halton_samples(bounds = self.corners, dims = self.N)
        for sample in samples:
            df, metadata = self.create_small_dataset(sample)
            print("Loaded " + str(len(df)) + " records")
            self.generate_powerset(df)
            print(self.visited)

    def main(self):
        self.test_exploration()

if __name__ == "__main__":
    ex = Exploration()
    ex.main()
