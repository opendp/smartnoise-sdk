import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def log_cluster(real, synthetic, G, verbose = False):
    """
    From "Generation and evaluation of synthetic patient data"
    (https://bmcmedresmethodol.biomedcentral.com/track/pdf/10.1186/s12874-020-00977-1.pdf)
    
    The log-cluster metric [39] is a measure of the similarity of the underlying latent structure of the real and
    synthetic datasets in terms of clustering.
    
    For a random partition of a data set into two groups of sizes Nr and Ns, 
    we would expect that, on average, Nr/(Nr + Ns) percent of the observations 
    in each cluster belong to group a.
    
    Note: we may want to include weights for cluster importance as an optional
    feature
    """
    combined_df_with_indicator = pd.concat([real, synthetic], axis=0, keys=[0, 1]).reset_index(level=[0]).rename(columns={'level_0': 'indicator'})
    
    no_indicator = combined_df_with_indicator.loc[:, combined_df_with_indicator.columns != "indicator"]
    kmeans = KMeans(n_clusters=G, random_state=0).fit(no_indicator)
    combined_df_with_indicator.loc[:,'labels'] = kmeans.labels_
    
    if verbose:
        print(combined_df_with_indicator)
    
    n = combined_df_with_indicator['indicator'].value_counts()

    n_r = n[0]
    n_s = n[1]
    
    c = n_r / (n_r + n_s)
    
    summed_over_G = 0
    all_label_counts = combined_df_with_indicator['labels'].value_counts()
    real_label_counts = combined_df_with_indicator.loc[combined_df_with_indicator['indicator'] == 0]['labels'].value_counts()
    
    if verbose:
        print(all_label_counts)
        print(real_label_counts)
    
    for label_j in range(G):
        n_j = all_label_counts[label_j]
        n_r_j = real_label_counts[label_j]

        if verbose:
            print(n_r_j)
            print(n_j)

        g_j = ((n_r_j / n_j) - c)**2
        
        summed_over_G += g_j
    
    U_c = np.log((1.0/G) * summed_over_G)
    
    if verbose:
        print(U_c)

    return U_c