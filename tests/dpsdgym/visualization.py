import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import conf

def plot_epsilon_tstr_accuracies(results_json=None):
    """
    Will produce a accuracy-by-epsilon graph from an artifact.json
    file.
    """
    if not results_json or results_json == "":
        results_json = 'artifact.json'

    with open(results_json, 'r') as f:
        results = json.load(f)
    colors = ['--r','--b','--g']
    for i,d in enumerate(conf.KNOWN_DATASETS):
        plt.figure(figsize=(12,8))
        for i,(n, _) in enumerate(conf.SYNTHESIZERS):
            y_tstr = []
            y_tsts = []
            y_trtr = []
            x = []
            for eps in results[d]['tstr_avg'][n]:
                
                tstr = results[d]['tstr_avg'][n][eps]
                tstr_avg = np.sum(tstr) / len(tstr)
                tstr_max = np.max(tstr)
                
                tsts = results[d]['tsts_sra'][n][eps]
                tsts_avg = np.sum(tsts) / len(tsts)
                tsts_max = np.max(tsts)
                
                if i == 0:
                    trtr = results[d]['trtr_sra'][n]
                    trtr_avg = np.sum(trtr) / len(trtr)
                    trtr_max = np.max(trtr)
                    y_trtr.append(trtr_max)
                
                y_tstr.append(tstr_max)
                y_tsts.append(tsts_max)
                x.append(float(eps))
            
            plt.plot(x, y_tstr,colors[i % len(colors)], label = str(n) + "_tstr")
            if i == 0:
                plt.plot(x, y_trtr,'--k', label = "trtr")
            plt.ylim(0.2,1.01)
            plt.xscale("log")
            plt.legend()
            plt.title("Synthesizer Comparison ML Classification")
            plt.xlabel("epsilon (log scale)")
            plt.ylabel("accuracy (avg. across 10 models)")
        plt.figure(i)
        plt.savefig("temp.png")
        plt.show()

if __name__ == "__main__":
    args = sys.argv
    file = None
    if len(args) > 1:
        file = args[1]

    plot_epsilon_tstr_accuracies(file)