# Finding rare events

We are interested in rare events, that is in the transition points of reactions or their surrounding stable energy minima.

We can formulate this as a sampler, that given a ML force field, can extract the saddle points and/or minima of the potential energy surface (PES).
Traditional sampling algorithms like metadynamics or NEB incur a high computational cost, because they query the energy function (the force field) many times. 
We propose to use a sampler that itself is a neural network, and can thus learn to sample across systems. 
The sampler, takes in the as description of the desired system as a set of atom types and periodic cell, and outputs atom coordinates. 
The sampler is trained on a loss of the ML force field, e.g. minimal forces to target saddle points.

## Repo Overview

### 1d example
- main https://github.com/BurgerAndreas/meta-sampler/blob/main/DEM/dem/energies/simple_test_function.py
- config1 https://github.com/BurgerAndreas/meta-sampler/blob/main/DEM/configs/energy/simple_test.yaml
- config2 without data https://github.com/BurgerAndreas/meta-sampler/blob/main/DEM/configs/experiment/simple_test_idem.yaml
- config2 with prior replay buffer = with data https://github.com/BurgerAndreas/meta-sampler/blob/main/DEM/configs/experiment/simple_test_pdem.yaml

### 2d Alanine Dipeptide (dihedral angles)
WIP by Andreas at https://github.com/BurgerAndreas/meta-sampler/tree/tps-1d/alanine_dipeptide

### N^3d Alanine Dipeptide (all atom)
WIP

### Surface + single atom adsorbant
WIP by Nikolaj

## Installation
get mamba (better than conda)
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

```bash
mamba create -n ms python=3.10 -y
mamba activate ms
pip install pyscf numpy==1.24.4 plotly kaleido scipy scikit-learn matplotlib==3.8.4 seaborn black tqdm joblib einops pandas ipykernel botorch
pip install torch
# pip install jax flax
```

