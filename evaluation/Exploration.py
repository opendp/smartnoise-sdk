# This file contains methods to set and scan a search space of datasets that we shall be using for DP evaluation. 
# We need to search databases, neighboring pairs and queries for running the DP predicate test. 
# We shall start a small 3-row databases to create our 12 neighboring pairs per database
# Then we shall use halton sequence to generate a set of such 3 row databases randomly in a 3-D log-space
import numpy as np
import pandas as pd
import os

class Exploration:
    def __init__(self, dataset_size = 3):
        self.dataset_size = dataset_size
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.csv_path = r'../service/datasets/evaluation'
        self.df, self.dataset_path, self.file_name = self.create_small_dataset()
        print("Loaded " + str(len(self.df)) + " records")
        self.N = len(self.df)
        self.visited = []
    
    def create_small_dataset(self, file_name = "small"):
        userids = list(range(1, self.dataset_size+1))
        userids = ["A" + str(user) for user in userids]
        usage = [10**i for i in range(0, self.dataset_size*2, 2)]
        df = pd.DataFrame(list(zip(userids, usage)), columns=['UserId', 'Usage'])
        
        # Storing the data as a CSV
        file_path = os.path.join(self.file_dir, self.csv_path, file_name + ".csv")
        df.to_csv(file_path, sep=',', encoding='utf-8', index=False)
        return df, file_path, file_name
    
    # Given a list of N records in a database, create a powerset of neighboring datasets by traversing the edges of database search graph
    # Perform DFS to traverse the database search graph
    # Convention of CSV names = <d1/d2>_<list of row indexes in d1>_<row index removed to create d2>.csv
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
                    d1_file_path = os.path.join(self.file_dir, self.csv_path , "d1_" + filename + ".csv")
                    d2_file_path = os.path.join(self.file_dir, self.csv_path , "d2_" + filename + ".csv")
                    d1.to_csv(d1_file_path, sep=',', encoding='utf-8', index=False)
                    d2.to_csv(d2_file_path, sep=',', encoding='utf-8', index=False)
                    self.visited.append(filename)
                    self.generate_powerset(d2)
                else:
                    continue

            return

    def main(self):
        self.generate_powerset(self.df)
        print(self.visited)

if __name__ == "__main__":
    ex = Exploration()
    ex.main()