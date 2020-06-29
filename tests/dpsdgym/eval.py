from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
#from xgboost import XGBRegressor


import numpy as np
import pandas as pd

SEED = None

KNOWN_DATASETS = ['nursery']

KNOWN_MODELS = [AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
               LogisticRegression, GaussianMixture, MLPClassifier, DecisionTreeClassifier,
               GaussianNB, BernoulliNB, MultinomialNB, RandomForestClassifier, ExtraTreesClassifier]

MODEL_ARGS = {
    'AdaBoostClassifier': {
        'random_state': SEED,
        'n_estimators': 100
    },
    'BaggingClassifier': {
        'random_state': SEED
    },
    'GradientBoostingClassifier': {
        'random_state': SEED,
        'n_estimators': 200
    },
    'LogisticRegression': {
        'random_state': SEED,
        'max_iter': 200,
        'multi_class': 'auto',
        'solver': 'lbfgs'
    },
    'GaussianMixture': {
        'random_state': SEED,
        'max_iter': 200
    },
    'MLPClassifier': {
        'random_state': SEED,
        'max_iter': 500
    },
    'DecisionTreeClassifier': {
        'random_state': SEED,
        'class_weight': 'balanced'
    },
    'GaussianNB': {
    },
    'BernoulliNB': {
    },
    'MultinomialNB': {
    },
    'RandomForestClassifier': {
        'random_state': SEED,
        'class_weight': 'balanced',
        'n_estimators': 200
    },
    'ExtraTreesClassifier': {
        'random_state': SEED,
        'class_weight': 'balanced',
        'n_estimators': 200
    }
}

# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVR
# gsc = GridSearchCV(
#         estimator=SVR(kernel='rbf'),
#         param_grid={
#             'C': [0.1, 1, 100, 1000],
#             'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
#             'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
#         },
#         cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

# from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator

# class DPSynthEstimatorHP(BaseEstimator):
#     """
#     We want to take in:
#     1. The model for our data synthesis.
#     2. The evaluation model for parameter selection.
#     """

def grid_search():
    pass
#     from sklearn.model_selection import GridSearchCV
#     gsc = GridSearchCV(
#         estimator=SVR(kernel='rbf'),
#         param_grid = {'Q':[400,1000],[]...}
#         cv = 5,
#         scoring = (custom scoring))
    
def tune_hyperparams():
    pass

def load_data(req_datasets=[]):
    import requests
    import io
    import json
    # Returns a dictionary of datasets
    # TODO: Add memory check to make sure to not overwhelm computer
    # TODO: Add safety checks here
    if not req_datasets:
        req_datasets = KNOWN_DATASETS

    with open('datasets.json') as j:
        dsets = j.read()
    archive = json.loads(dsets)

    loaded_datasets = {}

    def retrieve_dataset(dataset):
        r = requests.post(dataset['url'])
        if r.ok:
            data = r.content.decode('utf8')
            df = pd.read_csv(io.StringIO(data), names=dataset['columns'].split(','), index_col=False)
            return df

        raise "Unable to retrieve dataset: " + dataset

    def select_column(scol):
        # Zero indexed, inclusive
        return scol.split(',')

    def encode_categorical(df,dataset):
        from sklearn.preprocessing import LabelEncoder

        encoders = {}

        for column in select_column(dataset['categorical_columns']):
            encoders[column] = LabelEncoder()
            df[column] = encoders[column].fit_transform(df[column])

        return {"data": df, "target": dataset['target'], "name": dataset['name']}


    for d in req_datasets:
        df = retrieve_dataset(archive[d])
        encoded_df_dict = encode_categorical(df, archive[d])
        loaded_datasets[d] = encoded_df_dict

    # Return dictionary of pd dataframes
    return loaded_datasets

def run_synthesizers(datasets, synthesizers=[], epsilons=[1.0]):
    import time
    for n, s in synthesizers:
        print('Synthesizer: '+ str(n))
        for d in datasets:
            datasets[d][n] = {}
            synth_dict = datasets[d][n]
            for e in epsilons:
                start = time.time()
                # TODO: Set epsilons more elegantly
                s.epsilon = e
                sampled = s.fit_sample((datasets[d]["data"]))
                end = time.time() - start
                print(datasets[d]["name"] + ' finished in ' + str(end))
                # Add that synthesized dataset to the dict
                # Keyed with epsilon value
                synth_dict[str(e)] = sampled
    return datasets

def ml_eval(data_dict, synthesizers, epsilons, seed=42, test_size = 0.2):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    real = data_dict["data"]
    X = real.loc[:, real.columns != data_dict['target']]
    y = real.loc[:, real.columns == data_dict['target']]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    def model_tests(dd,n,model,cat,xt,yt,eps=None):
        predictions = model.predict(xt)
        model_dict = dd[type(model).__name__]
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
        # print('Accuracy ' + cat + ' ' + str(type(model).__name__) + ':' + str(model_cat['accuracy']))
        return model_cat['accuracy']
    
    for n, _ in synthesizers:
        trtrs = []
        tstrs = {}
        tstss = {}
        
        for model in KNOWN_MODELS:
            m_name = type(model()).__name__
            model_args = MODEL_ARGS[m_name]
            print(m_name)
            model_real = model(**model_args)
            model_real.fit(x_train, y_train.values.ravel())
            
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
                model_fake.fit(x_train_synth, y_train_synth.values.ravel())
                
                #Test the model
                tstr = model_tests(data_dict, n, model_fake, 'TSTR', x_test, y_test, str(e))
                tsts = model_tests(data_dict, n, model_fake, 'TSTS', x_test_synth, y_test_synth, str(e))
                tstss[str(e)].append(tsts)
                tstrs[str(e)].append(tstr)
            
        data_dict['trtr_sra'] = trtrs
        data_dict['tsts_sra'] = tstss
        data_dict['tstr_avg'] = tstrs
        
    return data_dict
            
def eval_data(data_dicts, synthesizers, epsilons):
    print(data_dicts)
    for d in data_dicts:
        for n,_ in synthesizers:
            for e in epsilons:
                wass = wasserstein_randomization(data_dicts[d][n][str(e)], data_dicts[d]['data'], 1000)
                data_dicts[d][n]['wasserstein_'+str(e)] = wass
            
                ml_eval(data_dicts[d], synthesizers, epsilons)

                sra_score = sra(data_dicts[d]['trtr_sra'],data_dicts[d]['tsts_sra'][str(e)])
                print(sra_score)
                if 'sra' not in data_dicts[d]:
                    data_dicts[d]['sra'] = {}
                data_dicts[d]['sra'][str(e)] = sra_score
        
    return data_dicts

def sra(R, S):
    k = len(R)
    sum_I = 0
    for i in range(k):
        R_vals = np.array([R[i]-rj if i != k else None for k, rj in enumerate(R)])
        S_vals = np.array([S[i]-sj if i != k else None  for k, sj in enumerate(S)])
        I = (R_vals[R_vals != np.array(None)] * S_vals[S_vals != np.array(None)])
        I[I >= 0] = 1
        I[I < 0] = 0
        sum_I += I
    return np.sum((1 / (k * (k-1))) * sum_I)

def wasserstein_randomization(d1, d2, iters):
    """
    Calculate wasserstein randomization test results
    "We propose a metric based on
    the idea of randomization inference (Basu, 1980; Fisher, 1935). 
    Each data point is randomly assigned to one of two
    data sets and the similarity of the resulting two distributions 
    is measured with the Wasserstein distance. Repeating this
    random assignment a great number of times (e.g. 100000 times) 
    provides an empirical approximation of the distances’
    null distribution. Similar to the pMSE ratio score we then 
    calculate the ratio of the measured Wasserstein distance and
    the median of the null distribution to get a Wasserstein distance 
    ratio score that is comparable across different attributes.
    Again a Wasserstein distance ratio score of 0 would indicate that 
    two marginal distributions are identical. Larger scores
    indicate greater differences between distributions."
    """
    from scipy.stats import wasserstein_distance
    import matplotlib.pyplot as plt
    # pip install pyemd
    # https://github.com/wmayner/pyemd
    from pyemd import emd_samples

    assert(len(d1) == len(d2))
    l_1 = len(d1)
    d3 = np.concatenate((d1,d2))
    distances = []
    for i in range(iters):
        np.random.shuffle(d3)
        n_1, n_2 = d3[:l_1], d3[l_1:]
        dist = emd_samples(n_1, n_2, bins='auto')
        distances.append(dist)
    plt.hist(distances, bins=25)
    plt.show()

    d_pd = pd.DataFrame(distances)
    print(d_pd.describe())

    """
    import numpy as np
    import pandas as pd

    d1 = pd.read_csv('nursery.csv')
    d2 = pd.read_csv('synthetic.csv')

    d1 = d1.drop(d1.columns[[0]], axis=1)
    d2 = d2.drop(d2.columns[[0]], axis=1)

    wasserstein_randomization(d1.to_numpy(), d2.to_numpy(), 1000)
    """
    



def run_suite(synthesizers=[], req_datasets=[], epsilons=[0.01, 0.1, 1.0, 10.0, 100.0]):
    import json
    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'to_json'):
                return {}
                #return obj.to_json(orient='records')
            return json.JSONEncoder.default(self, obj)
        
    loaded_datasets = load_data(req_datasets)
    data_dicts = run_synthesizers(loaded_datasets, synthesizers, epsilons)
    results = eval_data(data_dicts, synthesizers, epsilons)
    
    with open('artifact.json', 'w') as f:
        json.dump(results, f, cls=JSONEncoder)

        
"""
{
   "nursery":{
      "data":{},
      "target":"health",
      "name":"nursery",
      "mwem":{},
      "AdaBoostClassifier":{},
      "BaggingClassifier":{},
      "GradientBoostingClassifier":{},
      "LogisticRegression":{},
      "GaussianMixture":{},
      "MLPClassifier":{},
      "DecisionTreeClassifier":{},
      "GaussianNB":{},
      "BernoulliNB":{},
      "MultinomialNB":{},
      "RandomForestClassifier":{},
      "ExtraTreesClassifier":{},
      "trtr_sra":[],
      "tsts_sra":{},
      "tstr_avg":{},
      "sra":{}
   }
}
"""