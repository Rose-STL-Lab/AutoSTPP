<p align="center" >
  <a href="https://github.com/Rose-STL-Lab/AI-STPP"><img src="https://raw.githubusercontent.com/Rose-STL-Lab/AutoSTPP/refs/heads/main/Auto-STPP.png" width="256" height="256" alt="AI-STPP"></a>
</p>
<h1 align="center">Auto-STPP</h1>
<h4 align="center">✨Automatic Integration for Neural Spatiotemporal Point Process✨</h4>

<p align="center">
    <a href="https://raw.githubusercontent.com/Rose-STL-Lab/AutoSTPP/refs/heads/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="license"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-yellow" alt="python">
    <img src="https://img.shields.io/badge/Version-1.1.0-green" alt="version">
</p>

## | Introduction

**Auto**matic Integration for Neural **S**patio-**T**emporal **P**oint **P**rocess models (Auto-STPP) is a new paradigm for exact, efﬁcient, non-parametric inference of spatiotemporal point process.

## | Citation

[[2310.06179] Automatic Integration for Spatiotemporal Neural Point Processes](https://arxiv.org/abs/2310.06179)

```
@article{zhou2023automatic,
  title={Automatic Integration for Spatiotemporal Neural Point Processes},
  author={Zhou, Zihao and Yu, Rose},
  journal={arXiv preprint arXiv:2310.06179},
  year={2023}
}
```

## | Installation

Dependencies: `make`, `conda-lock`

```bash
make create_environment
conda activate autoint-stpp
```

## | Dataset Download

```bash
python src/download_data.py
```

## | Training and Testing

Specify the parameters in `configs/autoint_stpp.yaml` and then run

```bash
make run_stpp config=autoint_stpp
```
