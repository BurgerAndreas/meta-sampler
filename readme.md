# Finding rare events

We are interested in rare events, that is in the transition points of reactions or their surrounding stable energy minima.

We can formulate this as a sampler, that given a ML force field, can extract the saddle points and/or minima of the potential energy surface (PES).
Traditional sampling algorithms like metadynamics or NEB incur a high computational cost, because they query the energy function (the force field) many times. 
We propose to use a sampler that itself is a neural network, and can thus learn to sample across systems. 
The sampler, takes in the as description of the desired system as a set of atom types and periodic cell, and outputs atom coordinates. 
The sampler is trained on a loss of the ML force field, e.g. minimal forces to target saddle points.

Notes
- First try to overfit to a single example, i.e. force field only trained on Au-water. Then show that sampler can generalize across systems (i.e. MACE trained on many materials)
- First try to just e.g. minimize energy, then experiment with finding all extremal points
- Possibly experiment with auxiliary loss of forces=0 at extremal points?
- Saddle points or other point types of interest?

### Installation
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

