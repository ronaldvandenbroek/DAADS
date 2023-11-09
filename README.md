# Detecting Anomalies with Autoencoders on Data Streams 

This repository contains a replication of the results from "Detecting Anomalies with Autoencoders on Data Streams" (https://github.com/lucasczz/DAADS) by L. Cazzonelli and C. Kulbach.

## Installation
### Setup Git
```bash
git clone https://github.com/ronaldvandenbroek/DAADS.git
```
### Setup Environment
```bash
cd DAADS
```
```bash
conda create -n "daads_env" python=3.10
```
```bash
conda activate daads_env_cython
```
### Setup Dependancies
Install setuptools to enable cython to compile to C.
```bash
pip install --upgrade setuptools
```
```bash
pip install cython==0.29.36
``` 
```bash
pip install -r requirements.txt
```
For the plotting.ipynb notebook LaTeX is needed, if this is not this can be obtained with the following commands: 
```bash
sudo apt-get install aptitude
```
```bash
sudo aptitude install texlive-fonts-recommended texlive-fonts-extra
```
```bash
pip install -r requirements.txt
```
### Setup Path
Change [USER] into your local username. Depending on the virtual enviroment used the path might differ.\

This path is valid for Miniconda version 23.9.0.

```bash
LD_LIBRARY_PATH=:/home/[USER]/miniconda3/envs/daads_env_cython/lib/python3.10/site-packages/nvidia/cublas/lib/$LD_LIBRARY_PATH
```


## Reproducing the results
All experiment scripts are located in `./tools`.\
The experiment results are stored in `./results`.\
The notebooks to generate graphs and tables from the results are located in `./notebooks`.

### Evaluate all models
```bash
python ./tools/benchmark_exp.py
```
### Run contamination experiment
```bash
python ./tools/contamination_exp.py
```
### Run capacity experiment
```bash
python ./tools/capacity_exp.py
```
### Run learning rate experiment
```bash
python ./tools/lr_exp.py
```
### Generate the HST baseline
```bash
python ./tools/hst_exp.py
```
### Obtain anomaly scores
```bash
python ./tools/scores_exp.py
```
### Generating replicated plots
```
./notebooks/plotting.ipynb
```
### Generating comparison plots
```
./notebooks/additional_tests.ipynb
```

## Access datasets 
```shell 
from IncrementalTorch.datasets import Covertype, Shuttle
from river.datasets import CreditCard
```


