import subprocess
import os

import pytest
import warnings
import string
import numpy as np
import pandas as pd

from opendp.whitenoise.metadata import CollectionMetadata
from opendp.whitenoise.synthesizers.mwem import MWEMSynthesizer

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

meta_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.yaml"))
csv_path = os.path.join(git_root_dir, os.path.join("service", "datasets", "PUMS.csv"))

schema = CollectionMetadata.from_file(meta_path)
df = pd.read_csv(csv_path, index_col=0)
df = df.drop(["income"], axis=1)
nf = df.to_numpy().astype(int)

synth = MWEMSynthesizer(split_factor=3)

faux_synth = MWEMSynthesizer(split_factor=1)

test_data = np.array([[1,1,1],[2,2,2],[3,3,3]])

test_histogram = [[[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]],
        [[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]],
        [[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]]]

test_histogram_dims = (3,3,3)

class TestMWEM:
    def test_fit(self):
        synth.fit(nf)
        assert synth.histograms
    
    def test_sample(self):
        sample_size = nf.shape[0]
        synthetic = synth.sample(sample_size)
        assert synthetic.shape == nf.shape

    def test_initialize_A(self):
        h = synth._initialize_A(test_histogram,(3,3,3))
        assert int(np.sum(h)) == int(np.sum(test_histogram))
    
    def test_histogram_from_data_attributes(self):
        three_dims = synth._histogram_from_data_attributes(test_data,np.array([[0,1,2]]))
        one_dims = synth._histogram_from_data_attributes(test_data,np.array([np.array([0]),np.array([1]),np.array([2])]))
        assert three_dims[0][1] == [3,3,3]
        assert one_dims[0][1] == [3]

    def test_compose_arbitrary_slices(self):
        ss = synth._compose_arbitrary_slices(10, (3,3,3))
        assert np.array(ss).shape == (10,3)

    def test_evaluate(self):
        ss = synth._evaluate([slice(0, 2, None), slice(0, 2, None), slice(0, 3, None)],np.array(test_histogram))
        assert ss == 6.0

    def test_binary_replace_in_place_slice(self):
        b = synth._binary_replace_in_place_slice(np.array(test_histogram), [slice(0, 2, None), slice(0, 2, None), slice(0, 3, None)])
        assert (b == np.array([[[1., 1., 0.],
                            [1., 1., 0.],
                            [0., 0., 1.]],
                            [[1., 1., 0.],
                            [1., 1., 0.],
                            [0., 0., 1.]],
                            [[1., 1., 0.],
                            [1., 1., 0.],
                            [0., 0., 1.]]])).all()

    def test_reorder(self):
        original = np.array([[1,2,3,4,5,6], [6,7,8,9,10,11]])
        splits = np.array([[1,3,4],[0,2,5]])
        m1 = original[:, splits[0]]
        m2 = original[:, splits[1]]
        reordered = synth._reorder(splits)
        reconstructed = np.hstack((m1,m2))
        assert (original == reconstructed[:, reordered]).all()
    
    def test_generate_splits(self):
        assert (synth._generate_splits(3,3) == np.array([[0, 1, 2]])).all()

    # TODO: More split tests

    def test_faux_fit(self):
        pytest.warns(Warning, faux_synth.fit, test_data) 
        assert faux_synth.histograms