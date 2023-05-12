def test_real_vs_synthetic_data(real, synthetic, model, tsne=False, box=False, describe=False):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    synth_df = pd.DataFrame(synthetic, 
        columns=real.columns)

    X = real.iloc[:, :-1]
    y = real.iloc[:, -1]
    X_synth = synth_df.iloc[:, :-1]
    y_synth = synth_df.iloc[:, -1]
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train_synth, x_test_synth, y_train_synth, y_test_synth = train_test_split(X_synth, y_synth, test_size=0.2, random_state=42)
    
    model_real = model()
    model_real.fit(x_train, y_train)

    model_fake = model()
    model_fake.fit(x_train_synth, y_train_synth)
    
    #Test the model
    predictions = model_real.predict(x_test)
    print()
    print('Trained on Real Data')
    print(classification_report(y_test, predictions))
    print('Accuracy real: ' + str(accuracy_score(y_test, predictions)))
    
    predictions = model_fake.predict(x_test)
    print()
    print('Trained on Synthetic Data')
    print(classification_report(y_test, predictions))
    print('Accuracy synthetic: ' + str(accuracy_score(y_test, predictions)))

    # How does it compare to guessing randomly?
    print()
    print('Random Guessing')
    guesses = np.random.randint(0,(max(y_test_synth)-min(y_test_synth) + 1),len(y_test_synth))
    np.random.shuffle(guesses)
    print(classification_report(y_test_synth, guesses))
    print('Accuracy guessing: ' + str(accuracy_score(y_test_synth, guesses)))

    # TSNE Plot
    if tsne:
        from sklearn.manifold import TSNE
        comb = np.vstack((x_train[:500], x_train_synth[:500]))
        embedding_1 = TSNE(n_components=2, perplexity=5.0, early_exaggeration=1.0).fit_transform(comb)
        x,y = embedding_1.T
        l = int(len(x) / 2)
        inds = []

        plt.rcParams["figure.figsize"] = (15,15)
        plt.scatter(x,y,c=['purple' if i in inds else 'red' for i in range(l)]+['purple' if j in inds else 'blue' for j in range(l)])
        plt.gca().legend(('Real Data','Real'))
        plt.title('TSNE Plot, Real Data vs. Synthetic')
        plt.show()

    # Box plot
    if box:
        import math
        import seaborn as sns
        fig = plt.figure(figsize=(20,15))
        cols = 5
        rows = math.ceil(float(real.shape[1]) / cols)
        for i, column in enumerate(real.columns):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title(column)
            sns.boxplot(data=[real[column], synth_df[column]])
        plt.subplots_adjust(hspace=0.7, wspace=0.2)

    if describe:
        print(real.describe())
        print(synth_df.describe())

    return model_real, model_fake

