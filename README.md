<p align="center" >
  <a href="https://github.com/Rose-STL-Lab/AI-STPP"><img src="https://fremont.zzhou.info/images/2022/10/06/image-20221006102054441.png" width="256" height="256" alt="AI-STPP"></a>
</p>
<h1 align="center">AI-STPP</h1>
<h4 align="center">✨Automatic Integration for Neural Spatiotemporal Point Process✨</h4>

<p align="center">
    <a href="https://zzhou.info/LICENSE"><img src="https://camo.githubusercontent.com/87d0b0ec1c0a97dbf68ce4d3098de6912bca75aa006304dd0a55976e6673cbe1/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f64656c67616e2f6c6f677572752e737667" alt="license"></a>
    <img src="https://img.shields.io/badge/Python-3.8+-yellow" alt="python">
    <img src="https://img.shields.io/badge/Version-1.0.0beta1-green" alt="version">
</p>

## 丨Introduction

**A**utomatic **I**ntegration for Neural **S**patio-**T**emporal **P**oint **P**rocess models (AI-STPP) is a new paradigm for exact, efﬁcient, non-parametric inference of point process. It is capable of learning complicated underlying intensity functions, like a damped sine wave.

AI-STPP is under active development. Try the latest release version in the `main` branch, which include a working 1D case. We are working on the 3D case and trying to do some code refactoring and improvements. 


## | Comparison with State-of-the-Art
<details>
<summary>Learning Hawkes with AI-TPP or numerical integration</summary>
<img src="https://fremont.zzhou.info/images/2022/10/06/image-20221006104850204.png" alt="help">
</details>

<details>
<summary>Learning damped sine wave with AI-TPP or other baselines</summary>
<img src="https://fremont.zzhou.info/images/2022/10/06/image-20221006105113074.png" alt="help">
</details>

## 丨 Installation

`pip install -r requirements.txt`

## 丨 Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

