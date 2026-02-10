import numpy as np
import pandas as pd

from snsynth.models.dp_covariance import DPcovariance


def test_release_uses_clipped_data(monkeypatch):
    # Make Laplace noise exactly zero so output equals the internal covariance.
    monkeypatch.setattr(np.random, "uniform", lambda size: np.full(size, 0.5))

    data = pd.DataFrame({"x": [2.0, 0.0], "y": [2.0, 0.0]})
    bounds = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})

    model = DPcovariance(n=2, cols=["x", "y"], rng=bounds, global_eps=1.0)
    release = np.array(model.release(data))

    # Expected covariance of clipped data [[1,1],[0,0]].
    clipped = np.array([[1.0, 1.0], [0.0, 0.0]]).T
    cov = np.cov(clipped)
    expected = cov[np.tril_indices(cov.shape[0])]

    np.testing.assert_allclose(release, expected)
