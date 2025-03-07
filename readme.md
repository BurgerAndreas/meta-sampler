# Finding rare events

We are interested in rare events, that is in the transition points of reactions or their surrounding stable energy minima.

We can formulate this as a sampler, that given a ML force field, can extract the saddle points and/or minima of the potential energy surface (PES).
Traditional sampling algorithms like metadynamics or NEB incur a high computational cost, because they query the energy function (the force field) many times. 
We propose to use a sampler that itself is a neural network, and can thus learn to sample across systems. 
The sampler, takes in the as description of the desired system as a set of atom types and periodic cell, and outputs atom coordinates. 
The sampler is trained on a loss of the ML force field, e.g. minimal forces to target saddle points.

## Repo Overview

### 1d example
- main https://github.com/BurgerAndreas/meta-sampler/blob/main/dem/energies/simple_test_function.py
- config1 https://github.com/BurgerAndreas/meta-sampler/blob/main/configs/energy/simple_test.yaml
- config2 without data https://github.com/BurgerAndreas/meta-sampler/blob/main/configs/experiment/simple_test_idem.yaml
- config2 with prior replay buffer = with data https://github.com/BurgerAndreas/meta-sampler/blob/main/configs/experiment/simple_test_pdem.yaml

### 2d Alanine Dipeptide (dihedral angles)
WIP by Andreas at https://github.com/BurgerAndreas/meta-sampler/tree/tps-1d/alanine_dipeptide and

```bash
mamba activate sampler

# 1d toy example
python dem/train.py experiment=simple_test_idem

# double well
python dem/train.py experiment=dw4_idem
python dem/train.py experiment=dw4_pdem


########################################################

# 2d GMM (from the DEM paper)
python dem/train.py experiment=gmm_idem

# reduces the accuracy
python dem/train.py experiment=gmm_idem model.num_samples_to_sample_from_buffer=16 model.nll_batch_size=32 model.eval_batch_size=256 model.num_estimator_mc_samples=16

# saves accuracy with Richardson extrapolation
python dem/train.py experiment=gmm_idem model.num_samples_to_sample_from_buffer=16 model.nll_batch_size=32 model.eval_batch_size=256 model.num_estimator_mc_samples=16 model.use_richardsons=true

# TODO: doesn't work yet. saves accuracy by using a streaming estimator?
python dem/train.py experiment=gmm_idem model.num_samples_to_sample_from_buffer=16 model.nll_batch_size=32 model.eval_batch_size=256 model.streaming_batch_size=16 

# increasing num_estimator_mc_samples does improve accuracy
python dem/train.py experiment=gmm_idem model.num_samples_to_sample_from_buffer=16 model.nll_batch_size=32 model.eval_batch_size=256 

# increasing num_samples_to_sample_from_buffer does improve accuracy?
python dem/train.py experiment=gmm_idem model.nll_batch_size=32 model.eval_batch_size=256 model.num_estimator_mc_samples=16

###########################################################

# should give the same results as the original GMM
python dem/train.py experiment=gmm_idem_pseudo_test 

# 2d GMM with pseudo-energy
# My handcrafted AND Hessian
python dem/train.py experiment=gmm_idem_pseudo model.use_richardsons=true model.nll_batch_size=256

# double well
python dem/train.py experiment=dw4_idem model.use_richardsons=true 
python dem/train.py experiment=dw4_idem model.use_richardsons=true model.generate_constrained_samples=True


###########################################################

# 2d alanine dipeptide
python dem/train.py experiment=aldi_2d
```

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


