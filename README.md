<p align="center" >
  <a href="https://github.com/Rose-STL-Lab/AI-STPP"><img src="https://fremont.zzhou.info/images/2023/05/18/Auto-STPP.png" width="256" height="256" alt="AI-STPP"></a>
</p>
<h1 align="center">Auto-STPP</h1>
<h4 align="center">✨Automatic Integration for Neural Spatiotemporal Point Process✨</h4>

<p align="center">
    <a href="https://zzhou.info/LICENSE"><img src="https://camo.githubusercontent.com/87d0b0ec1c0a97dbf68ce4d3098de6912bca75aa006304dd0a55976e6673cbe1/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f64656c67616e2f6c6f677572752e737667" alt="license"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-yellow" alt="python">
    <img src="https://img.shields.io/badge/Version-1.1.0-green" alt="version">
</p>

## | Introduction

**Auto**matic Integration for Neural **S**patio-**T**emporal **P**oint **P**rocess models (Auto-STPP) is a new paradigm for exact, efﬁcient, non-parametric inference of spatiotemporal point process.

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
