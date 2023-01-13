import subprocess
import os

import numpy as np
import pandas as pd

from snsynth.mwem import MWEMSynthesizer
from snsynth.mwem import MWEMSynthesizer as ShortMWEMSynthesizer

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS.csv"))

df = pd.read_csv(csv_path, index_col=None)
df = df.drop(["income"], axis=1)
nf = df.to_numpy().astype(int)

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
    def test_short_import_works(self):
        assert MWEMSynthesizer == ShortMWEMSynthesizer

    def test_ranges(self):
        cont = df[['age', 'educ']]
        synth = MWEMSynthesizer(30.0, add_ranges=True)
        synth.fit(cont)
        rows = synth.sample(1000)
        age = np.mean(rows['age'])
        assert(age < 54)
        assert(age > 30)

    def test_fit(self):
        synth = MWEMSynthesizer(30.0, split_factor=3)
        synth.fit(nf)
        assert synth.histograms
        assert(np.isclose(synth.spent, synth.epsilon))

        # test sample
        sample_size = nf.shape[0]
        synthetic = synth.sample(sample_size)
        assert synthetic.shape == nf.shape

        age = np.mean(synthetic[:,0])
        assert(age < 54)
        assert(age > 30)

        # test sample different sizes with epsilon
        sample_size = int(nf.shape[0] / 2)
        synthetic = synth.sample(sample_size)
        assert synthetic.shape[0] == int(nf.shape[0] / 2)

    def test_sample_default_params(self):
        synth_default_params = MWEMSynthesizer()
        nf_slim = nf.copy()
        nf_slim = nf_slim[:,1:]
        synth_default_params.fit(nf_slim)
        sample_size = nf_slim.shape[0]
        synthetic = synth_default_params.sample(sample_size)
        print(synthetic[0:2,:])
        assert synthetic.shape == nf_slim.shape
        assert(np.isclose(synth_default_params.spent, synth_default_params.epsilon))
    
    def test_sample_different_sizes(self):
        synth_df = MWEMSynthesizer(3., split_factor=3, verbose=True)
        synth_df.fit(df)
        assert synth_df.histograms
        assert(np.isclose(synth_df.spent, synth_df.epsilon))

        sample_size = int(df.shape[0] / 2)
        synthetic = synth_df.sample(sample_size)
        assert synthetic.shape[0] == int(df.shape[0] / 2)

        sample_size = int(nf.shape[0] * 2)
        synthetic = synth_df.sample(sample_size)
        assert synthetic.shape[0] == int(df.shape[0] * 2)

    def test_initialize_a(self):
        synth = MWEMSynthesizer(3., split_factor=3)
        h = synth._initialize_a(test_histogram,(3,3,3))
        assert int(np.sum(h.data)) == int(np.sum(test_histogram))

    def test_histogram_from_data_attributes(self):
        synth = MWEMSynthesizer(3., split_factor=3)
        three_dims = synth._histogram_from_data_attributes(test_data,np.array([[0,1,2]]))
        one_dims = synth._histogram_from_data_attributes(test_data,np.array([np.array([0]),np.array([1]),np.array([2])]))
        assert three_dims[0].dimensions == [3,3,3]
        assert one_dims[0].dimensions == [3]

    def test_reorder(self):
        synth = MWEMSynthesizer(3., split_factor=3)
        original = np.array([[1,2,3,4,5,6], [6,7,8,9,10,11]])
        splits = np.array([[1,3,4],[0,2,5]])
        m1 = original[:, splits[0]]
        m2 = original[:, splits[1]]
        reordered = synth._reorder(splits)
        reconstructed = np.hstack((m1,m2))
        assert (original == reconstructed[:, reordered]).all()

    def test_generate_splits(self):
        synth = MWEMSynthesizer(3., split_factor=3)
        assert (synth._generate_splits(3,3) == np.array([[0, 1, 2]])).all()

