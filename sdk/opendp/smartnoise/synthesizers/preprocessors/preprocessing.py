import pandas as pd
import numpy as np


def get_metadata(data, categorical_columns=tuple(), ordinal_columns=tuple()):
    meta = []

    df = pd.DataFrame(data)
    for col_name in df:
        column = df[col_name]

        if categorical_columns and col_name in categorical_columns:
            bins = column.value_counts().index.tolist()
            meta.append({"name": col_name, "type": "categorical", "size": len(bins), "bins": bins})
        elif ordinal_columns and col_name in ordinal_columns:
            value_count = list(dict(column.value_counts()).items())
            value_count = sorted(value_count, key=lambda x: -x[1])
            bins = list(map(lambda x: x[0], value_count))
            meta.append({"name": col_name, "type": "ordinal", "size": len(bins), "bins": bins})
        else:
            meta.append(
                {"name": col_name, "type": "continuous", "min": column.min(), "max": column.max()}
            )

    return meta


class GeneralTransformer:
    def __init__(self, n_clusters=10, eps=0.005):
        """n_cluster is the upper bound of modes."""
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = get_metadata(data, categorical_columns, ordinal_columns)
        model = []

        self.output_info = []
        self.output_dim = 0
        self.components = []
        for id_, info in enumerate(self.meta):
            if info["type"] == "continuous":
                raise Exception("Use of BayesianGaussianMixture for continuous variables is "
                                "being evaluated to avoid privacy leaks. "
                                "Until resolved, 'continuous' columns are not supported with the GeneralTransformer.")
            else:
                model.append(None)
                self.components.append(None)
                self.output_info += [(info["size"], "softmax")]
                self.output_dim += info["size"]

        self.model = model

    def transform(self, data):
        values = []
        for id_, info in enumerate(self.meta):
            current = data.iloc[:, id_]
            if info["type"] == "continuous":
                current = current.values.reshape([-1, 1])

                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = (current - means) / (4 * stds)

                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))

                n_opts = sum(self.components[id_])
                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                opt_sel = np.zeros(len(data), dtype="int")
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -0.99, 0.99)

                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                values += [features, probs_onehot]
            else:
                col_t = np.zeros([len(data), info["size"]])
                idx = list(map(info["bins"].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data, sigmas):
        data_t = np.zeros([len(data), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info["type"] == "continuous":
                u = data[:, st]
                v = data[:, st + 1:st + 1 + np.sum(self.components[id_])]

                if sigmas is not None:
                    sig = sigmas[st]
                    u = np.random.normal(u, sig)

                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * 100
                v_t[:, self.components[id_]] = v
                v = v_t
                st += 1 + np.sum(self.components[id_])
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.nanargmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 4 * std_t + mean_t
                data_t[:, id_] = tmp

            else:
                current = data[:, st:st + info["size"]]
                st += info["size"]
                idx = np.nanargmax(current, axis=1)
                data_t[:, id_] = list(map(info["bins"].__getitem__, idx))

        return data_t
