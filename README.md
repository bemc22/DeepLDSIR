# Deep Low-Dimensional Spectral Image Representation for Compressive Spectral Reconstruction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hdspgroup/DeepLDSIR/blob/master/demo_recons.ipynb)
[![DOI:10.1109/MLSP52302.2021.9596541](https://zenodo.org/badge/DOI/10.1109/MLSP52302.2021.9596541.svg)](https://doi.org/10.1109/MLSP52302.2021.9596541)

## Abstract

Model-based deep learning techniques are the state-of-the-art in compressive spectral imaging reconstruction. These methods integrate deep neural networks (DNN) as spectral image representation used as prior information in the optimization problem, showing optimal results at the expense of increasing the dimensionality of the non-linear representation, i.e., the number of parameters to be recovered. This paper proposes an autoencoder-based network that guarantees a low-dimensional spectral representation through feature reduction, which can be used as prior in the compressive spectral imaging reconstruction. Additionally, based on the experimental observation that the obtained low dimensional spectral representation preserves the spatial structure of the scene, this work exploits the sparsity in the generated feature space by using the Wavelet basis to reduce even more the dimensionally of the inverse problem. The proposed method shows improvements up to 2 dB against state-of-the-art methods.

## Colab Demo

| Notebook      | Link          |
| ------------- | ------------- |
| Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hdspgroup/DeepLDSIR/blob/master/demo_train.ipynb)  |
| Reconstruction  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hdspgroup/DeepLDSIR/blob/master/demo_recons.ipynb)  |


## How to cite
If this code is useful for your and you use it in an academic work, please consider citing this paper as


```bib
@INPROCEEDINGS{9596541,
  author={Monroy, Brayan and Bacca, Jorge and Arguello, Henry},
  booktitle={2021 IEEE 31st International Workshop on Machine Learning for Signal Processing (MLSP)}, 
  title={Deep Low-Dimensional Spectral Image Representation for Compressive Spectral Reconstruction}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/MLSP52302.2021.9596541}}
```
