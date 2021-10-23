# Whitepaper Demo Notebooks

[<img src="/whitepaper-demos/images/income-distribution.png" alt="histogram of income distribution" height="100">](/whitepaper-demos/3-histogram-sample.ipynb)
[<img src="/whitepaper-demos/images/QUAIL.png" alt="quail" height="100">](/whitepaper-demos/5-ml-synthetic-data.ipynb)
[<img src="/whitepaper-demos/images/auc-score-dp-naive-bayes.png" alt="quail" height="100">](/whitepaper-demos/4-ml-dp-classifier.ipynb)
[<img src="/whitepaper-demos/images/synthetic.jpg" alt="synthetic dataset" height="100">](/whitepaper-demos/5-ml-synthetic-data.ipynb)

The following notebooks are based on the whitepaper titled [Microsoft SmartNoise Differential Privacy Machine Learning Case Studies](https://azure.microsoft.com/en-us/resources/microsoft-smartnoisedifferential-privacy-machine-learning-case-studies/).

## Python Notebooks 
The following notebooks use the SmartNoise SDK through Python bindings.

- [Protecting against Reidentification Attacks with Differential Privacy](https://github.com/opendp/smartnoise-samples/blob/master/whitepaper-demos/2-reidentification-attack.ipynb):  Demonstration of how differential privacy can be used to protect sensitive personal information against re-identification attacks. 
  - Notebook: [`2-reidentification-attack.ipynb`](https://github.com/opendp/smartnoise-samples/blob/master/whitepaper-demos/2-reidentification-attack.ipynb)
- [Privacy-Preserving Statistical Analysis with Differential Privacy](https://github.com/opendp/smartnoise-samples/blob/master/whitepaper-demos/3-histogram-sample.ipynb): Generate and release basic statistical outcomes in a differentially private fashion using demographic data about Californian residents from the Public Use Microdata Sample (PUMS) statistics. 
  - Notebook: [`3-histogram-sample.ipynb`](https://github.com/opendp/smartnoise-samples/blob/master/whitepaper-demos/3-histogram-sample.ipynb)
- [Privacy Preserving Machine Learning with Differential Privacy](https://github.com/opendp/smartnoise-samples/blob/master/whitepaper-demos/4-ml-dp-classifier.ipynb): The goal of this notebook is to demonstrate how to perform supervised machine learning on a tabular data with differential privacy. This ensures that the contribution of the individuals' data to the resulting machine learning model is masked out.
  -  Notebook: [`4-ml-dp-classifier.ipynb`](https://github.com/opendp/smartnoise-samples/blob/master/whitepaper-demos/4-ml-dp-classifier.ipynb)
- [Using SmartNoise to create synthetic data with high utility for machine learning](https://github.com/opendp/smartnoise-samples/blob/master/whitepaper-demos/5-ml-synthetic-data.ipynb): This notebook demonstrates how to use SmartNoise to create a synthetic dataset, which can then be used to achieve comparable performance to other differentially private options for machine learning. The essential advantage of the synthesizer approach is that the differentially private dataset can be analyzed any number of times without increasing the privacy risk.
  -  Notebook: [`5-ml-synthetic-data.ipynb`](https://github.com/opendp/smartnoise-samples/blob/master/whitepaper-demos/5-ml-synthetic-data.ipynb)
- [Privacy Preserving Deep Learning for Medical Image Analysis](https://github.com/opendp/smartnoise-samples/blob/master/whitepaper-demos/6-deep-learning-medical.ipynb): The goal of this notebook is to demonstrate how to perform deep learning for medical image analysis in conjunction with differential privacy. This ensures that the contribution of the individuals (patients in this case) to the resulting machine learning model is masked out.
  - Notebook: [`6-deep-learning-medical.ipynb`](https://github.com/opendp/smartnoise-samples/blob/master/whitepaper-demos/6-deep-learning-medical.ipynb)
