## Samples for MWEMSynthesizer, and scaled implementation of:
HARDT, M., LIGETT, K., AND MCSHERRY, F. 2012. A simple and practical algorithm for differentially private data release. arXiv:1012.4763 (https://www.cs.huji.ac.il/~katrina//papers/mwem-nips.pdf)

Note distinction between this implementation and the one released in the paper: in order to scale MWEMSynthesizer, we provide the ability to divide a database into "splits" and run the algorithm on these split databases with a suitably divided budget. We sample from the split databases without merging the "factored" distributions (as is described in the paper), and combine the samples to produce a full sample from the original database. This offers significant performance benefits for local experiments.

## Recommended order for navigating MWEM samples, from view first to view last:

### 1. Visualizing MWEM
Gives a sense of the underlying functionality of MWEM via visualizations of private vs. synthetic histograms.

### 2. Car Dataset MWEM (Multiclass)
This is an ideal dataset for MWEM: purely categorical. MWEM is able to draw from a distribution across all the data and produce differentially private synthetic data that performs comparably to the original data on the multiclass task in a matter of seconds.
    
### 3. Adult Dataset Classification (Binary)
This is a difficult dataset for MWEM: combined categorical, ordinal and continuous. We must "split" the database into independent distributions, and sample from those, before recombining into a final distribution. We see that MWEM is still able to provide high quality synthetic data, although this data is not well suited for the adult dataset classification problem.

### 4. MNIST MWEM (Image Data)
This is a near impossible dataset for MWEM: image data. Despite splitting the database in such a way that we jeopardize privacy, and using some of MWEMs most invasive tuning mechanisms, we find that the data we generate is not of very high quality. This example serves to show the limits of MWEMs capabilities.
