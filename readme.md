# Transition State Sampling

Chemical reactions are the result of a transition from reactants to products.
The rate of an reaction depends on the activation energy, which is the energy barrier that must be overcome for the reaction to occur.
The transition state is the point of highest energy on the potential energy surface (PES) between reactants and products.
Thus knowing all transitions states and minima of system determines all possible reactions and their frequency.

Transitions states also appear in protein folding between different local energy minima.

Transitions are rare events, and thus seldomly observed, which makes naive simulation of transitions states computationally expensive, as measured by number of times they query the energy function. 
Traditional sampling algorithms are limited in that (1) they do not amortize (learn) across systems (2) limited exploration due to high cost to escape local regions (metadynamics) or already require knowledge of minima as starting points (NEB, TPS).
These problems are usually excagerated in higher dimensions.

Instead of performing simulation or local optimization, we directly generate the transition states.
Since ground truth transition states are not known, we do not use training data, but instead use a (twice/three times differentiable) energy function (e.g. ML force field).
We propose to use a diffusion-based neural sampler, that samples the unnormalized Boltzmann distribution of a pseudo- energy function that is minimal at transition states.
The sampler takes as input the desired system as a set of atom types (and periodic cell), and outputs atom coordinates. The sampler can thus learn to sample transition states across systems.



## Usage

```bash
mamba activate sampler

# double well (single transition state)
python dem/train.py experiment=dw_idem
python dem/train.py experiment=dw_idem_condforce

# four wells (multiple transition states)
python dem/train.py experiment=fw_idem
python dem/train.py experiment=fw_idem_condforce

# 2d alanine dipeptide (dihedral angles as collective variables)
python dem/train.py experiment=aldi2d_idem
```

Run on the cluster
```bash
ssh ...@comps0.cs.toronto.edu

mamba activate sampler
sbatch scripts/cslab_launcher.slrm train.py experiment=aldi2d_idem energy.temperature=3000 +addon=mem2
```

N^3d Alanine Dipeptide (all atom)
WIP by Andreas

Surface + single atom adsorbant
WIP by Nikolaj


## Installation

Get your favourite package manager (I like mamba)
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

mamba create -n sampler python=3.11
mamba activate sampler
pip install --upgrade pip setuptools wheel
```

Install dependencies
```bash
pip install setuptools==59.2.0 numpy==1.24.4 scikit-learn plotly kaleido imageio scipy matplotlib seaborn black tqdm joblib einops ipykernel toml omegaconf nbformat nglview py3Dmol hydra-core==1.*
mamba install openmm==8.2.0 -y
```

Install PyTorch and PyTorch Geometric (adjust to your CUDA version)
```bash
# First uninstall any existing installations
pip uninstall torch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -y

# Install PyTorch for CUDA 12.6
pip install --no-index torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --no-index pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric
```

Install MACE
```bash
# for fast mace (5-9x speedup)
pip uninstall mace-torch -y
# mace==0.3.10 was designed to work with cuequivariance==0.1.0
cd ..
pip install cuequivariance==0.1.0 cuequivariance-torch cuequivariance-ops-torch-cu12
# git clone https://github.com/ACEsuit/mace
git clone https://github.com/BurgerAndreas/vectorizable-mace.git
pip install vectorizable-mace/.
cd meta-sampler
``` 

Install Boltzmann Generator Utils
```bash
pip install mdtraj==1.9.9 
# pip install mdtraj==1.9.8
mamba install mdtraj=1.9.9 -y
pip install tables
pip install git+https://github.com:noegroup/bgmol.git
```

Install DEM sampler
```bash
pip install lightning==2.* torchmetrics==0.* hydra-core==1.* rich==13.* pre-commit==3.* pytest==7.* wandb hydra-optuna-sweeper hydra-colorlog rootutils normflows nflows einops torchsde torchdiffeq torchcfm numpy==1.24.4
pip install git+https://github.com/jarridrb/fab-torch.git
pip install git+https://github.com/VincentStimper/resampled-base-flows.git git+https://github.com/jarridrb/fab-torch.git git+https://github.com/atong01/bgflow.git  
pip install -U numpy==1.24.4

pip install -e .

export WANDB_ENTITY=<...>
```

~~Modify Mace and e3nn library according to `mace.md`~~

```bash
pip uninstall e3nn -y
pip install git+https://github.com/BurgerAndreas/vectorizable-e3nn.git
```

## Acknowledgements

The code heavily builds on iDEM (Iterated Denoising Energy Matching for Sampling from Boltzmann Densities)
[(Preprint)](https://arxiv.org/abs/2402.06121) [(Code)](https://github.com/jarridrb/dem)


### Citations

If this codebase is useful towards other research efforts please consider citing us.

```
@misc{akhoundsadegh2024iterated,
      title={Iterated Denoising Energy Matching for Sampling from Boltzmann Densities},
      author={Tara Akhound-Sadegh and Jarrid Rector-Brooks and Avishek Joey Bose and Sarthak Mittal and Pablo Lemos and Cheng-Hao Liu and Marcin Sendera and Siamak Ravanbakhsh and Gauthier Gidel and Yoshua Bengio and Nikolay Malkin and Alexander Tong},
      year={2024},
      eprint={2402.06121},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


### Licences

This repo is licensed under the [MIT License](https://opensource.org/license/mit/).

