import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score

from metrics.sra import sra
from metrics.wasserstein import wasserstein_randomization
from metrics.pmse import pmse_ratio

from load_data import load_data

import conf 

import numpy as np
import pandas as pd

class dumb_predictor():
    """
    Dummy classifier to be used if any of conf.KNOWN_MODELS break.
    Returns single class as prediction.
    """
    def __init__(self, label):
        self.label = label
        
    def predict(self, instances):
        return np.full(len(instances), self.label)

def run_synthesizers(datasets, synthesizers=[], epsilons=[1.0]):
    """
    Run each synthesizer on each dataset for specified epsilons

    :param datasets: dictionary with real datasets
    :type datasets: dict
    :param synthesizers: list of synthesizers used, often KNOWN_SYNTHESIZERS
    :type synthesizers: list
    :param epsilons: Epsilons used in the synthesis
    :type epsilons: list
    :return: dictionary of both real and synthetic data for each dataset
    :rtype: dict
    """
    import time
    for n, s in synthesizers:
        print('Synthesizer: '+ str(n))
        for d in datasets:
            datasets[d][n] = {}
            synth_dict = datasets[d][n]
            synth_args = conf.SYNTH_SETTINGS[n][d]
            for e in epsilons:
                start = time.time()
                # TODO: Set epsilons more elegantly
                synth = s(epsilon=float(e), **synth_args)
                sampled = synth.fit_sample((datasets[d]["data"]))
                end = time.time() - start
                print(datasets[d]["name"] + ' finished in ' + str(end))
                # Add that synthesized dataset to the dict
                # Keyed with epsilon value
                synth_dict[str(e)] = sampled
    return datasets

def model_tests(dd, n, model, cat, xt, yt, eps=None):
    """
    Helper function for ml_eval(), used to examine 
    performance post model fit.
    """
    predictions = model.predict(xt)
    if n not in dd[type(model).__name__]:
        dd[type(model).__name__][n] = {}
    model_dict = dd[type(model).__name__][n]
    if eps:
        if cat not in model_dict:
            model_dict[cat] = {}
        model_dict[cat][eps] = {}
        model_cat = model_dict[cat][eps]
    else:
        model_dict[cat] = {}
        model_cat = model_dict[cat]
    model_cat['classification_report'] = classification_report(np.ravel(yt), predictions, labels=np.unique(predictions))
    model_cat['accuracy'] = accuracy_score(np.ravel(yt), predictions)
    
    # Check to see if classification problem is multiclass
    if type(model).__name__ != 'dumb_predictor':
        probs = model.predict_proba(xt)
        unique = np.array(np.unique(yt))
        if len(unique) > 2:
            try:
                model_cat['aucroc'] = roc_auc_score(yt, probs, multi_class='ovr')
            except:
                try:
                    # We can try again, removing classes that have no
                    # examples in yt
                    missing_classes = np.setdiff1d(model.classes_, unique)
                    cols = []
                    for m in missing_classes:
                        ind = np.where(model.classes_==m)
                        cols.append(ind[0].tolist())
                        
                    existant_probs = np.delete(probs, cols, axis=1)

                    for i, row in enumerate(existant_probs):
                        existant_probs[i] = row / np.sum(row)
                        
                    model_cat['aucroc'] = roc_auc_score(yt, existant_probs, multi_class='ovr')
                except:
                    # If this doesnt work, we admit defeat
                    model_cat['aucroc'] = 0.0
        else:
            if len(np.unique(yt)) == 1:
                # Occasionally, the synthesizer will
                # produce synthetic labels of only one class
                # - in this case, aucroc is undefined, so we set 
                # it to 0 (undesirable)
                model_cat['aucroc'] = 0.0
            else:
                probs = probs[:,0]
                model_cat['aucroc'] = roc_auc_score(yt, probs)
            
    return model_cat['accuracy']

def ml_eval(data_dict, synthesizers, epsilons, seed=42, test_size = 0.2):
    """
    Takes in output from load_data.py/run_synthesizers().
    Trains each model in conf.KNOWN_MODELS with real/fake data,
    and compares performance. Records necessary accuracies to
    be used with SRA.

    :param data_dicts: dictionary with real/synthetic data
    :type data_dicts: dict
    :param synthesizers: list of synthesizers used, often KNOWN_SYNTHESIZERS
    :type synthesizers: list
    :param epsilons: Epsilons used in the synthesis
    :type epsilons: list
    :return: dictionary artifact.json with all results
    :rtype: dict
    """
    real = data_dict["data"]
    X = real.loc[:, real.columns != data_dict['target']]
    y = real.loc[:, real.columns == data_dict['target']]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    for n, _ in synthesizers:
        trtrs = []
        tstrs = {}
        tstss = {}

        for model in conf.KNOWN_MODELS:
            m_name = type(model()).__name__
            model_args = conf.MODEL_ARGS[m_name]
            print(m_name)
            model_real = model(**model_args)
            model_real.fit(x_train, y_train.values.ravel())
            
            if type(model_real).__name__ not in data_dict:
                data_dict[type(model_real).__name__] = {}
            
            trtr = model_tests(data_dict, n, model_real, 'TRTR', x_test, y_test)
            trtrs.append(trtr)
            
            for e in epsilons:
                if str(e) not in tstss:
                    tstss[str(e)] = []
                if str(e) not in tstrs:
                    tstrs[str(e)] = []
                    
                synth = data_dict[n][str(e)]
                
                X_synth = synth.loc[:, synth.columns != data_dict['target']]
                y_synth = synth.loc[:, synth.columns == data_dict['target']]
                x_train_synth, x_test_synth, y_train_synth, y_test_synth = train_test_split(X_synth, y_synth, test_size=test_size, random_state=seed)

                model_fake = model(**model_args)
                y_train_ravel = y_train_synth.values.ravel()
                if len(np.unique(y_train_ravel)) > 1:
                    model_fake.fit(x_train_synth, y_train_ravel)
                    #Test the model
                    tstr = model_tests(data_dict, n, model_fake, 'TSTR', x_test, y_test, str(e))
                    tsts = model_tests(data_dict, n, model_fake, 'TSTS', x_test_synth, y_test_synth, str(e))
                    tstss[str(e)].append(tsts)
                    tstrs[str(e)].append(tstr)
                elif len(np.unique(y_train_ravel)) == 1:
                    dumb = dumb_predictor(np.unique(y_train_ravel)[0])
                    data_dict[type(dumb).__name__] = {}
                    tstr = model_tests(data_dict, n, dumb, 'TSTR', x_test, y_test, str(e))
                    tsts = model_tests(data_dict, n, dumb, 'TSTS', x_test_synth, y_test_synth, str(e))
                    tstss[str(e)].append(tsts)
                    tstrs[str(e)].append(tstr)

        if 'trtr_sra' not in data_dict:
            data_dict['trtr_sra'] = {}
        if 'tsts_sra' not in data_dict:
            data_dict['tsts_sra'] = {}
        if 'tstr_avg' not in data_dict:
            data_dict['tstr_avg'] = {}
        data_dict['trtr_sra'][n] = trtrs
        data_dict['tsts_sra'][n] = tstss
        data_dict['tstr_avg'][n] = tstrs
        
    return data_dict
            
def eval_data(data_dicts, synthesizers, epsilons, flags):
    """
    Takes in output from load_data.py/run_synthesizers(). Runs flagged 
    metrics for each dataset specified.

    :param data_dicts: dictionary with real/synthetic data
    :type data_dicts: dict
    :param synthesizers: list of synthesizers used, often KNOWN_SYNTHESIZERS
    :type synthesizers: list
    :param epsilons: Epsilons used in the synthesis
    :type epsilons: list
    :param flags: The metrics to run
    :type flags: list
    :return: dictionary artifact.json with all results
    :rtype: dict
    """
    for d in data_dicts:
        for n,_ in synthesizers:
            for e in epsilons:
                if 'wasserstein' in flags:
                    wass = wasserstein_randomization(data_dicts[d][n][str(e)], data_dicts[d]['data'], 1000)
                    data_dicts[d][n]['wasserstein_'+str(e)] = wass

                if 'pmse' in flags:
                    pmse = pmse_ratio(data_dicts[d][n][str(e)], data_dicts[d]['data'])
                    data_dicts[d][n]['pmse_'+str(e)] = pmse

                if 'ml_eval' in flags:
                    ml_eval(data_dicts[d], synthesizers, epsilons)

                    if 'sra' in flags:
                        sra_score = sra(data_dicts[d]['trtr_sra'][n],data_dicts[d]['tsts_sra'][n][str(e)])
                        print(sra_score)
                        if 'sra' not in data_dicts[d]:
                            data_dicts[d][n]['sra'] = {}
                        data_dicts[d][n]['sra'][str(e)] = sra_score
        
    return data_dicts

    
def run_suite(synthesizers=[], req_datasets=[], epsilons=[0.01, 0.1, 1.0, 9.0, 45.0, 95.0], flags=[]): 
    import json
    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'to_json'):
                return {}
                # TODO: Provide flagged option for saving the
                # datasets themselves. Current default to
                # discarding data after run
                # return obj.to_json(orient='records')
            return json.JSONEncoder.default(self, obj)
        
    loaded_datasets = load_data(req_datasets)
    data_dicts = run_synthesizers(loaded_datasets, synthesizers, epsilons)
    results = eval_data(data_dicts, synthesizers, epsilons, flags)
    
    with open('artifact.json', 'w') as f:
        json.dump(results, f, cls=JSONEncoder)

flag_options = ['wasserstein', 'ml_eval', 'sra', 'pmse']

if __name__ == "__main__":
    # TODO: Add epsilon flag to specify epsilons pre run
    args = sys.argv

    if args[1] == 'all' or args == None:
        flags = flag_options
    else: 
        flags = args[1:]

    run_suite(synthesizers=conf.SYNTHESIZERS, flags=flags)