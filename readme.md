# Meta sampler

New and better force fields are published by the week. Time to consider MLFF as a new data modality.

We want a sampler, that given a ML force field, can extract the minima and maxima of the potential energy surface (PES). Traditional sampling algorithms like metadynamics or NEB incur a high computational cost, because they query the energy function (the force field) many times. We propose to use a sampler that itself is a neural network, and can thus learn to sample across systems. The sampler acts as a meta network, taking in the weights of a ML force fields and the desired system as a set of atom types, and outputs atom coordinates. The sampler is trained to minimize/maximize the predicted energy of the ML force field. 

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

