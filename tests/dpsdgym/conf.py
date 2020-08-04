
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from opendp.whitenoise.synthesizers.mwem import MWEMSynthesizer
from opendp.whitenoise.synthesizers.quail import QUAILSynthesizer
from opendp.whitenoise.synthesizers.superquail import SuperQUAILSynthesizer
from opendp.whitenoise.synthesizers.pytorch.pytorch_synthesizer import PytorchDPSynthesizer
from opendp.whitenoise.synthesizers.preprocessors.preprocessing import GeneralTransformer
from opendp.whitenoise.synthesizers.pytorch.nn import DPGAN, PATEGAN
from dpctgan import DPCTGANSynthesizer

from diffprivlib.models import LogisticRegression as DPLR
from diffprivlib.models import GaussianNB as DPGNB

SEED = 42

KNOWN_DATASETS =  ['mushroom'] # ,'adult', 'wine', 'car', 'nursery'

KNOWN_MODELS = [AdaBoostClassifier, BaggingClassifier,
               LogisticRegression, MLPClassifier,
               RandomForestClassifier]

KNOWN_MODELS_STR = ['AdaBoostClassifier', 'BaggingClassifier',
               'LogisticRegression', 'MLPClassifier',
               'GaussianNB', 'RandomForestClassifier']

SYNTHESIZERS = [
    ('mwem', MWEMSynthesizer),
    # ('dpctgan', DPCTGANSynthesizer),
    # ('dpgan',PytorchDPSynthesizer),
    # ('pategan',PytorchDPSynthesizer),
    # ('quail_mwem', QUAILSynthesizer),
    # ('quail_dpgan', QUAILSynthesizer),
    # ('quail_pategan', QUAILSynthesizer),
    # ('quail_dpctgan', QUAILSynthesizer),
]

SYNTH_SETTINGS = {
    'superquail': {
        'adult': {
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            }
        },
        'car': {
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            }
        },
        'wine': {
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            }
        },
        'mushroom': {
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            }
        }
    },
    'dpctgan': {
        'adult': {
            'epochs': 50
        },
        'car': {
            'epochs': 50
        },
        'wine': {
            'epochs': 50
        },
        'mushroom': {
            'epochs': 50
        }
    },
    'quail_dpctgan': {
        'adult': {
            'dp_synthesizer': DPCTGANSynthesizer,
            'synth_args': {
                'epochs': 50
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'earning-class'
        },
        'car': {
            'dp_synthesizer': DPCTGANSynthesizer,
            'synth_args': {
                'epochs': 50
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'class'
        },
        'wine': {
            'dp_synthesizer': DPCTGANSynthesizer,
            'synth_args': {
                'epochs': 50
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'quality'
        },
        'mushroom': {
            'dp_synthesizer': DPCTGANSynthesizer,
            'synth_args': {
                'epochs': 50
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'edible'
        },
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
    'quail_dpgan': {
        'car': {
            'dp_synthesizer': PytorchDPSynthesizer,
            'synth_args': {
                'preprocessor': GeneralTransformer(),
                'gan': DPGAN(batch_size=640, epochs=100)
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'class'
        },
        'wine': {
            'dp_synthesizer': PytorchDPSynthesizer,
            'synth_args': {
                'preprocessor': GeneralTransformer(),
                'gan': DPGAN(batch_size=640, epochs=100)
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'quality'
        },
        'mushroom': {
            'dp_synthesizer': PytorchDPSynthesizer,
            'synth_args': {
                'preprocessor': GeneralTransformer(),
                'gan': DPGAN(batch_size=640, epochs=100)
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'edible'
        },
        'adult': {
            'dp_synthesizer': PytorchDPSynthesizer,
            'synth_args': {
                'preprocessor': GeneralTransformer(),
                'gan': DPGAN(batch_size=640, epochs=100)
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'earning-class'
        },
    },
    'quail_pategan': {
        'car': {
            'dp_synthesizer': PytorchDPSynthesizer,
            'synth_args': {
                'preprocessor': GeneralTransformer(),
                'gan': PATEGAN(batch_size=640)
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'class'
        },
        'wine': {
            'dp_synthesizer': PytorchDPSynthesizer,
            'synth_args': {
                'preprocessor': GeneralTransformer(),
                'gan': PATEGAN(batch_size=640)
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'quality'
        },
        'mushroom': {
            'dp_synthesizer': PytorchDPSynthesizer,
            'synth_args': {
                'preprocessor': GeneralTransformer(),
                'gan': PATEGAN(batch_size=640)
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'edible'
        },
        'adult': {
            'dp_synthesizer': PytorchDPSynthesizer,
            'synth_args': {
                'preprocessor': GeneralTransformer(),
                'gan': PATEGAN(batch_size=1280)
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'earning-class'
        },
    },
    'quail_mwem': {
        'nursery': {
            'dp_synthesizer': MWEMSynthesizer,
            'synth_args': {
                'Q_count':1000,
                'iterations':30,
                'mult_weights_iterations':20,
                'split_factor':8,
                'max_bin_count':400
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'health'
        },
        'car': {
            'dp_synthesizer': MWEMSynthesizer,
            'synth_args': {
                'Q_count':400,
                'iterations':20,
                'mult_weights_iterations': 15,
                'split_factor':7,
                'max_bin_count':400
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'class'
        },
        'mushroom': {
            'dp_synthesizer': MWEMSynthesizer,
            'synth_args': {
                'Q_count':1000,
                'iterations':30,
                'mult_weights_iterations':20,
                'split_factor':4,
                'max_bin_count':400
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'edible'
        },
        'wine': {
            'dp_synthesizer': MWEMSynthesizer,
            'synth_args': {
                'Q_count':400,
                'iterations':25,
                'mult_weights_iterations':15,
                'split_factor':3,
                'max_bin_count':200
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'quality'
        },
        'adult': {
            'dp_synthesizer': MWEMSynthesizer,
            'synth_args': {
                'Q_count': 400,
                'iterations':20,
                'mult_weights_iterations':15,
                'splits':[[0,1,2],[3,4,5],[6,7,8],[9,10],[11,12],[13,14]],
                'max_bin_count':100
            },
            'dp_classifier': DPLR,
            'class_args': {
                'max_iter': 1000
            },
            'target': 'earning-class'
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