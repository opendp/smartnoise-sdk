import pandas as pd


def test_real_vs_synthetic_data(real, synthetic: pd.DataFrame, test_data, model,
                               categorical_features,
                                tsne=False):
    import numpy as np
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt

    X_real = real.iloc[:, :-1]
    y_real = real.iloc[:, -1]
    X_synth = synthetic.iloc[:, :-1]
    y_synth = synthetic.iloc[:, -1]

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    model_real = model()
    model_real.fit(X_real, y_real)

    model_fake = model()
    model_fake.fit(X_synth, y_synth)
    
    #Test the model
    predictions = model_real.predict(X_test)
    print()
    print('Trained on Real Data')
    print(classification_report(y_test, predictions))
    print('Accuracy real: ' + str(accuracy_score(y_test, predictions)))
    
    predictions = model_fake.predict(X_test)
    print()
    print('Trained on Synthetic Data')
    print(classification_report(y_test, predictions))
    print('Accuracy synthetic: ' + str(accuracy_score(y_test, predictions)))

    # TSNE Plot
    if tsne:
        from sklearn.manifold import TSNE
        comb = np.vstack((X_real[:500], X_synth[:500]))
        embedding_1 = TSNE(n_components=2, perplexity=5.0, early_exaggeration=1.0).fit_transform(comb)
        x,y = embedding_1.T
        l = int(len(x) / 2)
        inds = []

        plt.rcParams["figure.figsize"] = (15,15)
        plt.scatter(x,y,c=['purple' if i in inds else 'red' for i in range(l)]+['purple' if j in inds else 'blue' for j in range(l)])
        plt.gca().legend(('Real Data','Real'))
        plt.title('TSNE Plot, Real Data vs. Synthetic')
        plt.show()

    return model_real, model_fake

