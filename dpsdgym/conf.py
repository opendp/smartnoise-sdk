from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from snsynth.mwem import MWEMSynthesizer
from snsynth.pytorch.pytorch_synthesizer import PytorchDPSynthesizer
from snsynth.preprocessors.preprocessing import GeneralTransformer
from snsynth.pytorch.nn import DPGAN, PATEGAN, DPCTGAN, PATECTGAN

from diffprivlib.models import LogisticRegression as DPLR
from diffprivlib.models import GaussianNB as DPGNB

# Keep seed consistent for reproducibility
SEED = 42

# Turn on/off balancing imbalanced data with SMOTE
BALANCE = True

# Turn on/off the synthesizers you want to use in eval here
SYNTHESIZERS = [
    ('mwem', MWEMSynthesizer),
    ('dpctgan', PytorchDPSynthesizer),
    ('patectgan', PytorchDPSynthesizer),
    ('dpgan',PytorchDPSynthesizer),
    ('pategan',PytorchDPSynthesizer),
]

# Define the defaults epsilons you want to use in eval
EPSILONS = [0.01, 0.1, 0.5, 1.0, 3.0, 6.0, 9.0]

# Add datasets on which to evaluate synthesis
KNOWN_DATASETS =  ['bank','adult','mushroom','shopping','car']

# Default metrics used to evaluate differential privacy 
KNOWN_METRICS = ['wasserstein', 'ml_eval', 'pmse', 'sra']

# Add ML models on which to evaluate utility
KNOWN_MODELS = [AdaBoostClassifier, BaggingClassifier,
               LogisticRegression, MLPClassifier,
               RandomForestClassifier]

# Mirror strings for ML models, to log
KNOWN_MODELS_STR = ['AdaBoostClassifier', 'BaggingClassifier',
               'LogisticRegression', 'MLPClassifier',
               'GaussianNB', 'RandomForestClassifier']

SYNTH_SETTINGS = {
    'dpctgan': {
        'default': {
            'gan': DPCTGAN(epsilon=0.1, epochs=100)
        }
    },
    'patectgan': {
        'default': {
            'gan': PATECTGAN(epsilon=0.1, epochs=100)
        },
    },
    'dpgan': {
        'default': {
            'preprocessor': GeneralTransformer(),
            'gan': DPGAN(epsilon=0.1, batch_size=640, epochs=100)
        },
    },
    'pategan': {
        'default': {
            'preprocessor': GeneralTransformer(),
            'gan': PATEGAN(epsilon=0.1, batch_size=1280)
        }
    },
    'mwem': {
        'car': {
            'q_count':400,
            'iterations':20,
            'mult_weights_iterations': 15,
            'split_factor':7,
            'max_bin_count':400
        },
        'mushroom': {
            'q_count':400,
            'iterations':30,
            'mult_weights_iterations':20,
            'split_factor':4,
            'max_bin_count':400
        },
        'bank': {
            'q_count':400,
            'iterations':25,
            'mult_weights_iterations':15,
            'split_factor':3,
            'max_bin_count':200
        },
        'adult': {
            'q_count': 400,
            'iterations':20,
            'mult_weights_iterations':15,
            'splits':[[0,1,2],[3,4,5],[6,7,8],[9,10],[11,12],[13,14]],
            'max_bin_count':100
        },
        'shopping': {
            'q_count':400,
            'iterations':30,
            'mult_weights_iterations':20,
            'split_factor':2,
            'max_bin_count':400
        },
        'default': {
            'q_count':400,
            'iterations':30,
            'mult_weights_iterations':20,
            'split_factor':3,
            'max_bin_count':400
        },
    }
}

MODEL_ARGS = {
    'AdaBoostClassifier': {
        'random_state': SEED,
        'n_estimators': 100
    },
    'BaggingClassifier': {
        'random_state': SEED
    },
    'LogisticRegression': {
        'random_state': SEED,
        'max_iter': 1000,
        'multi_class': 'auto',
        'solver': 'lbfgs'
    },
    'MLPClassifier': {
        'random_state': SEED,
        'max_iter': 2000,
        'early_stopping': True,
        'n_iter_no_change': 20
    },
    'DecisionTreeClassifier': {
        'random_state': SEED,
        'class_weight': 'balanced'
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
