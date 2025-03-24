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

### N^3d Alanine Dipeptide (all atom)
WIP

### Surface + single atom adsorbant
WIP by Nikolaj

## Installation

Get your favourite package manager (I like mamba)
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

mamba create -n sampler python=3.11
mamba activate sampler
pip install --upgrade pip setuptools wheel

pip install numpy==1.24.4 scikit-learn plotly kaleido imageio scipy matplotlib seaborn black tqdm joblib einops ipykernel toml omegaconf nbformat openmm nglview py3Dmol hydra-core==1.*
```

Install PyTorch and PyTorch Geometric (adjust to your CUDA version)
```bash
# First uninstall any existing installations
pip uninstall torch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -y

# Install PyTorch for CUDA 12.6
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Finally install torch-geometric
pip install torch-geometric

# alternative
# mamba install pytorch pytorch-cuda=12.6 -c pytorch -c nvidia
# mamba install pyg -c pyg

# for fast mace (5-9x speedup)
pip uninstall mace-torch -y
# mace==0.3.10 was designed to work with cuequivariance==0.1.0
pip install cuequivariance==0.1.0 cuequivariance-torch cuequivariance-ops-torch-cu12
# git clone https://github.com/ACEsuit/mace
git clone https://github.com/BurgerAndreas/vectorizable-mace.git
pip install vectorizable-mace/.
cd meta-sampler
```

Install Boltzmann Generator Utils
```bash
pip install mdtraj==1.9.9 tables

cd ..
git clone git@github.com:noegroup/bgmol.git
cd bgmol
python setup.py install
cd ..

cd meta-sampler
```

Install DEM sampler
```bash
pip install -u numpy==1.24.4
pip install setuptools==59.8.0
pip install lightning==2.* torchmetrics==0.* hydra-core==1.* rich==13.* pre-commit==3.* pytest==7.* wandb hydra-optuna-sweeper hydra-colorlog rootutils normflows nflows einops torchsde torchdiffeq torchcfm 
pip install git+https://github.com/VincentStimper/resampled-base-flows.git git+https://github.com/jarridrb/fab-torch.git git+https://github.com/atong01/bgflow.git

pip install -e .
```

~~Modify Mace and e3nn library according to `mace.md`~~

```bash
pip install git+https://github.com/BurgerAndreas/vectorizable-e3nn.git
```


---

<div align="center">

## Iterated Denoising Energy Matching for Sampling from Boltzmann Densities

[![Preprint](http://img.shields.io/badge/paper-arxiv.2402.06121-B31B1B.svg)](https://arxiv.org/abs/2402.06121)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

### Description

This is the official repository for the paper [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://arxiv.org/abs/2402.06121).

We propose iDEM, a scalable and efficient method to sample from unnormalized probability distributions. iDEM makes use of the DEM objective, inspired by the stochastic regression and simulation
free principles of score and flow matching objectives while allowing one to learn off-policy, in a loop while itself generating (optionally exploratory) new states which are subsequently
learned on. iDEM is also capable of incorporating symmetries, namely those represented by the product group of $SE(3) \\times \\mathbb{S}\_n$. We experiment on a 2D GMM task as well as a number of physics-inspired problems. These include:

- DW4 -- the 4-particle double well potential (8 dimensions total)
- LJ13 -- the 13-particle Lennard-Jones potential (39 dimensions total)
- LJ55 -- the 55-particle Lennard-Jones potential (165 dimensions total)

This code was taken from an internal repository and as such all commit history is lost here. Development credit for this repository goes primarily to [@atong01](https://github.com/atong01), [@jarridrb](https://github.com/jarridrb) and [@taraak](https://github.com/taraak) who built
out most of the code and experiments with help from [@sarthmit](https://github.com/sarthmit) and [@msendera](https://github.com/msendera). Finally, the code is based off the
[hydra lightning template](https://github.com/ashleve/lightning-hydra-template) by [@ashleve](https://github.com/ashleve) and makes use of the [FAB torch](https://github.com/lollcat/fab-torch) code for the GMM task and replay buffers.

### Installation

For installation, we recommend the use of Micromamba. Please refer [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for an installation guide for Micromamba.
First, we install dependencies

```bash
# clone project
git clone git@github.com:jarridrb/DEM.git
cd DEM

# create micromamba environment
micromamba create -f environment.yaml
micromamba activate dem

# install requirements
pip install -r requirements.txt

```

Note that the hydra configs interpolate using some environment variables set in the file `.env`. We provide
an example `.env.example` file for convenience. Note that to use wandb we require that you set WANDB_ENTITY in your
`.env` file.

To run an experiment, e.g., GMM with iDEM, you can run on the command line

```bash
python dem/train.py experiment=gmm_idem
```

We include configs for all experiments matching the settings we used in our paper for both iDEM and pDEM except LJ55 for
which we only include a config for iDEM as pDEM had convergence issues on this dataset.


### CFM for Computing NLL Pipeline

We will use the example of LJ55 in detailing the pipeline. First, run the training script as normal as follows

```bash
python dem/train.py experiment=lj55_idem
```

After training is complete, find the epochs with the best `val/2-Wasserstein` values in wandb. We will use the
best checkpoint to generate a training dataset for CFM in the following command. This command will also log the
2-Wasserstein and total variation distance for the dataset generated from the trained iDEM model compared to the
test set. To run this, you must provide the eval script with the checkpoint path you are using.

```bash
python dem/eval.py experiment=lj55_idem ckpt_path=<path_to_ckpt>
```

This will take some time to run and will generate a file named `samples_<n_samples_to_generate>.pt` in the hydra
runtime directory for the eval run. We can now use these samples to train a CFM model. We provide a config `lj55_idem_cfm`
which has the settings to enable the CFM pipeline to run by default for the LJ55 task, though doing so for other tasks
is also simple. The main config changes required are to set `model.debug_use_train_data=true, model.nll_with_cfm=true`
and `model.logz_with_cfm=true`. To point the CFM training run to the dataset generated from iDEM samples we can set the
`energy.data_path_train` attribute to the path of the generated samples. CFM training in this example can then be done
with

```bash
python dem/train.py experiment=lj55_idem_cfm energy.data_path_train=<path_to_samples>
```

Finally, to eval test set NLL, take the checkpoint of the CFM run with the best `val/nll` and run the eval script
again

```bash
python dem/eval.py experiment=lj55_idem_cfm ckpt_path=<path_to_cfm_ckpt>
```

Finally, we note that you may need to try a couple different checkpoints from the original
`python dem/train.py experiment=lj55_idem` run to be used in generating samples and downstream CFM training/eval in
order to get the best combination of eval metrics.

### ESS Computation Considerations

In preparing this update we noticed our original evaluation of ESS was evaluated on a batch size of 16 on all tasks. We recommend users of our
repository instead evaluate ESS on a larger batch size, (default to 1000) in the updated code. To reproduce the results in the paper you can
either set this to 16 or look at the wandb during validation when training the CFM model which evaluates on batch size 16.

### LJ55 negative time

In our original manuscript for LJ55 we used 10 steps of "negative time" (described in Section 4 of our manuscript)
during inference where we continued SDE inference for 10 extra steps using the true score at time 0. The repository
code had the flag to do this turned on in the configs but the code ignored this flag. This has been corrected in the update.

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

