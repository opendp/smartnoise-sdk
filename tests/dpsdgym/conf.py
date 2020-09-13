
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from opendp.whitenoise.synthesizers.mwem import MWEMSynthesizer
from opendp.whitenoise.synthesizers.pytorch.pytorch_synthesizer import PytorchDPSynthesizer
from opendp.whitenoise.synthesizers.preprocessors.preprocessing import GeneralTransformer
from opendp.whitenoise.synthesizers.pytorch.nn import DPGAN, PATEGAN, DPCTGAN

from diffprivlib.models import LogisticRegression as DPLR
from diffprivlib.models import GaussianNB as DPGNB

# Keep seed consistent for reproducibility 
SEED = 42

# Turn on/off the synthesizers you want to use in eval here
SYNTHESIZERS = [
    ('mwem', MWEMSynthesizer),
    ('dpctgan', PytorchDPSynthesizer),
    ('dpgan', PytorchDPSynthesizer),
]

# Add datasets on which to evaluate synthesis
KNOWN_DATASETS =  ['mushroom'] # ,'adult', 'wine', 'car', 'nursery'

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
        'adult': {
            'preprocessor': GeneralTransformer(),
            'gan': DPGAN(batch_size=640, epochs=100)
        },
        'car': {
            'preprocessor': GeneralTransformer(),
            'gan': DPGAN(batch_size=640, epochs=100)
        },
        'wine': {
            'preprocessor': GeneralTransformer(),
            'gan': DPGAN(batch_size=640, epochs=100)
        },
        'mushroom': {
            'preprocessor': GeneralTransformer(),
            'gan': DPGAN(batch_size=640, epochs=100)
        }
    },
    'dpgan': {
        'car': {
            'preprocessor': GeneralTransformer(),
            'gan': DPGAN(batch_size=640, epochs=100)
        },
        'wine': {
            'preprocessor': GeneralTransformer(),
            'gan': DPGAN(batch_size=640, epochs=100)
        },
        'mushroom': {
            'preprocessor': GeneralTransformer(),
            'gan': DPGAN(batch_size=640, epochs=100)
        },
        'adult': {
            'preprocessor': GeneralTransformer(),
            'gan': DPGAN(batch_size=640, epochs=100)
        }
    },
    'pategan': {
        'car': {
            'preprocessor': GeneralTransformer(),
            'gan': PATEGAN(batch_size=640)
        },
        'wine': {
            'preprocessor': GeneralTransformer(),
            'gan': PATEGAN(batch_size=640)
        },
        'mushroom': {
            'preprocessor': GeneralTransformer(),
            'gan': PATEGAN(batch_size=640)
        },
        'adult': {
            'preprocessor': GeneralTransformer(),
            'gan': PATEGAN(batch_size=1280)
        }
    },
    'mwem': {
        'nursery': {
            'Q_count':400,
            'iterations':30,
            'mult_weights_iterations':20,
            'split_factor':8,
            'max_bin_count':400
        },
        'car': {
            'Q_count':400,
            'iterations':20,
            'mult_weights_iterations': 15,
            'split_factor':7,
            'max_bin_count':400
        },
        'mushroom': {
            'Q_count':400,
            'iterations':30,
            'mult_weights_iterations':20,
            'split_factor':4,
            'max_bin_count':400
        },
        'wine': {
            'Q_count':400,
            'iterations':25,
            'mult_weights_iterations':15,
            'split_factor':3,
            'max_bin_count':200
        },
        'adult': {
            'Q_count': 400,
            'iterations':20,
            'mult_weights_iterations':15,
            'splits':[[0,1,2],[3,4,5],[6,7,8],[9,10],[11,12],[13,14]],
            'max_bin_count':100
        }
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
