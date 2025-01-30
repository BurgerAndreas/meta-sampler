# Sampling the potential energy surface of a PaiNN MLFF

We overfit to a single PaiNN MLFF, targeting minimal energy

### Install

```bash
git clone git@github.com:Yangxinsix/painn-sli.git
cd painn-sli

# 320 MB
# https://data.dtu.dk/articles/dataset/Dataset_for_Neural_Network_Potentials_for_Accelerated_Metadynamics_of_Oxygen_Reduction_Kinetics_at_Au-Water_Interfaces_/22284514?file=39631924
wget https://data.dtu.dk/ndownloader/files/39631924 -O dtudataset
```

```bash
# get mamba package manager
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

mamba create -n painn python=3.10
mamba activate painn
pip uninstall painn -y
pip install -e .

pip install pyscf numpy==1.24.4 plotly kaleido scipy matplotlib==3.8.4 seaborn black tqdm joblib einops ipykernel toml 

# First uninstall any existing installations
pip uninstall torch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -y

# Install PyTorch for CUDA 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Finally install torch-geometric
pip install torch-geometric

# alternative
# mamba install pytorch pytorch-cuda=12.6 -c pytorch -c nvidia
# mamba install pyg -c pyg
```

### Fit sampler (ours)
```bash
python scripts/train_sampler.py --load_model trained_models/96_node_3_layer.pth --batch_size 4
```

### Run PaiNN (previous)
* See `train.py` in `scripts` for training, and `md_run.py` for running MD simulations (Metadynamics) by using ASE.

```bash
python scripts/train.py
python scripts/train.py --load_model trained_models/96_node_3_layer.pth --batch_size 8

# run MD / metadynamics
python scripts/md_run.py --load_model trained_models/96_node_3_layer.pth
python scripts/md_run.py --load_model trained_models/96_node_3_layer.pth --plumed
```

## Citation

This is based on the source code for paper [Neural Network Potentials for Accelerated Metadynamics of Oxygen Reduction Kinetics at Au-Water Interfaces](https://pubs.rsc.org/en/content/articlelanding/2023/sc/d2sc06696c) 

