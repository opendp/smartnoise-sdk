import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import conf

def plot_epsilon_tstr_accuracies(dataset_dict):
    """
    Will produce a accuracy-by-epsilon graph from an artifact.json
    file.
    """
    colors = ['--r','--b','--g', '--c', '--m', '--y']
    for i,d in enumerate(dataset_dict):
        plt.figure(figsize=(12,8))
         