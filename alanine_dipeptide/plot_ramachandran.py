import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
import plotly.express as px
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
import scipy.special
import math

from openmm.app import PDBFile, ForceField, Simulation, NoCutoff, HBonds
from openmm import Context, VerletIntegrator, Platform
from openmm.unit import nanometer, kilojoule_per_mole, picoseconds, Quantity
import openmm
import bgmol
from bgmol.api import system_by_name

from mace.calculators import mace_off, mace_anicc
from alanine_dipeptide_openmm_amber99 import (
    compute_energy_and_forces_openmm,
    pdbfile,
    fffile,
)
from alanine_dipeptide_mace import (
    update_alanine_dipeptide_with_grad,
    update_alanine_dipeptide_with_grad_batched,
    load_alanine_dipeptide_ase,
    fffile,
    _atoms_to_batch,
)
from dihedral import (
    set_dihedral_torch,
    set_dihedral_torch_batched,
    set_dihedral,
    compute_dihedral_torch,
    compute_dihedral_torch_batched,
    compute_dihedral,
    update_neighborhood_graph_batched,
)

# from torch_cluster import radius_graph
from mace_neighbourshood import get_neighborhood


def compute_ramachandran_openmm_amber(
    phi_values,
    psi_values,
    recompute=False,
    convention="andreas",
):
    # if file exists, load it
    resolution = len(phi_values)
    datafile = f"alanine_dipeptide/outputs/ramachandran_openmm_amber_{resolution}_{convention}.npy"
    if os.path.exists(datafile) and not recompute:
        energies, forces_norm, forces_normmean = np.load(datafile)
    else:

        npdtype = np.float128
        torchdtype = torch.float64

        # Create a Ramachandran plot
        pdb = PDBFile(pdbfile)
        # Extract positions as a NumPy array in nanometers.
        positions_default = np.array(pdb.positions.value_in_unit(nanometer))

        # Create the force field.
        forcefield = ForceField(fffile)

        # Build the simulation / system
        # system = forcefield.createSystem(
        #     pdb.topology,
        #     removeCMMotion=True,
        #     constraints=HBonds,
        #     rigidWater=True,
        #     nonbondedMethod=NoCutoff,
        # )
        # integrator = VerletIntegrator(0.001 * picoseconds)
        # platform = Platform.getPlatformByName("Reference")
        # simulation = Simulation(pdb.topology, system, integrator, platform)

        temperature = 300
        system = system_by_name("AlanineDipeptideImplicit")
        integrator = openmm.LangevinIntegrator(temperature, 1, 0.001)
        simulation = openmm.app.Simulation(system.topology, system.system, integrator)

        simulation.context.setPositions(positions_default)
        state = simulation.context.getState(getEnergy=True)
        print(
            f"Initial energy: {state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)} kJ/mol"
        )

        # Initialize an array to store energies.
        energies = np.zeros((len(phi_values), len(psi_values)))
        forces_norm = np.zeros((len(phi_values), len(psi_values)))
        forces_normmean = np.zeros((len(phi_values), len(psi_values)))

        # Loop over grid points.
        for i, phi in tqdm(enumerate(phi_values), total=len(phi_values)):
            for j, psi in enumerate(psi_values):
                # Angles already in radians
                dihedrals = torch.tensor([phi, psi], dtype=torch.float32)

                # Set the positions.
                positions = positions_default.copy()
                positions = set_dihedral(positions, "phi", phi, "phi", convention)
                positions = set_dihedral(positions, "psi", psi, "psi", convention)

                # # build system from scratch
                # system = forcefield.createSystem(
                #     pdb.topology,
                #     removeCMMotion=True,
                #     constraints=HBonds,
                #     rigidWater=True,
                #     nonbondedMethod=NoCutoff,
                # )
                # integrator = VerletIntegrator(0.001 * picoseconds)
                # platform = Platform.getPlatformByName("Reference")
                # simulation = Simulation(pdb.topology, system, integrator, platform)

                # Set the positions.
                simulation.context.setPositions(positions)

                # Retrieve the state with energy and forces.
                state = simulation.context.getState(getEnergy=True, getForces=True)
                energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
                forces = state.getForces(asNumpy=True).value_in_unit(
                    kilojoule_per_mole / nanometer
                )

                # Convert energy and forces to torch tensors.
                energy = torch.tensor(energy, dtype=torchdtype)
                force = torch.tensor(forces, dtype=torchdtype)

                # Store the energy (in kJ/mol)
                energies[i, j] = energy.item()
                # force are shape [22,3] each row is a force vector for an atom
                forces_norm[i, j] = torch.linalg.norm(force).item()
                forces_normmean[i, j] = torch.linalg.norm(force, axis=1).mean().item()

                # print the computed energy for each grid point.
                # tqdm.write(
                #     f"phi={phi:6.3f}, psi={psi:6.3f} -> U={energy.item():8.2f} kJ/mol, F={forces_norm[i, j]:8.2f}"
                # )

        # Save the energies to a file
        np.save(datafile, (energies, forces_norm, forces_normmean))
    return energies, forces_norm, forces_normmean


def compute_ramachandran_mace(
    phi_values,
    psi_values,
    recompute=False,
    convention="andreas",
    batch_size=128,
):
    """
    Compute the Ramachandran plot for the alanine dipeptide energy landscape using the MACE model.
    Takes forces as derivative w.r.t. dihedral angles.
    """
    # if file exists, load it
    resolution = len(phi_values)
    datafile = (
        f"alanine_dipeptide/outputs/ramachandran_mace_{resolution}_{convention}.npy"
    )
    if os.path.exists(datafile) and not recompute:
        energies, forces_norm, forces_normmean = np.load(datafile)
    else:
        # get alanine dipeptide atoms
        atoms = load_alanine_dipeptide_ase()

        # Get MACE force field: mace_off or mace_anicc
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_off(model="medium", device=device_str)  # enable_cueq=True
        # calc = mace_anicc(device=device_str)  # enable_cueq=True
        device = calc.device
        atoms.calc = calc
        atoms_calc = atoms.calc

        ################################################
        # ASE atoms -> torch batch
        batch_base = atoms_calc._atoms_to_batch(atoms)

        if atoms_calc.model_type in ["MACE", "EnergyDipoleMACE"]:
            batch = atoms_calc._clone_batch(batch_base)
            node_heads = batch["head"][batch["batch"]]
            num_atoms_arange = torch.arange(batch["positions"].shape[0])
            node_e0 = atoms_calc.models[0].atomic_energies_fn(batch["node_attrs"])[
                num_atoms_arange, node_heads
            ]
            compute_stress = not atoms_calc.use_compile
        else:
            compute_stress = False
        batch_base = batch_base.to_dict()
        model = atoms_calc.models[0]

        ############################################################################
        # Compute energies / forces in minibatches
        ############################################################################
        energies_batched = []
        forces_norm_batched = []
        forces_normmean_batched = []
        positions_rotated_batched = []

        # Create grid of all phi/psi combinations
        phi_psi_grid = torch.cartesian_prod(
            torch.tensor(phi_values), torch.tensor(psi_values)
        )

        # compute in minibatches of batch_size
        num_atoms = batch_base["positions"].shape[0]
        num_edges = batch_base["edge_index"].shape[1]
        num_samples = phi_psi_grid.shape[0]
        num_batches = math.ceil(num_samples / batch_size)
        for i in range(num_batches):
            # last batch can be truncated (not enough samples left to fill the batch)
            bs = min(batch_size, num_samples - i * batch_size)
            # Make minibatch version of batch, that is just multiple copies of the same batch
            # but mimicks a batch from a typical torch_geometric dataloader
            minibatch_base = _atoms_to_batch(atoms_calc, atoms, bs=bs, repeats=bs)
            # n_atoms = torch.tensor([torch.sum(minibatch["batch"] == i) for i in range(bs)])

            phi_psi_batch = phi_psi_grid[i : i + bs].requires_grad_(True)

            # Update positions
            minibatch = update_alanine_dipeptide_with_grad_batched(
                phi_psi_batch, minibatch_base, convention=convention
            )

            # Update edge indices
            # TODO: why is the cell so huge?
            minibatch = update_neighborhood_graph_batched(
                minibatch, model.r_max.item(), overwrite_cell=True
            )

            # Compute energies and forces for all configurations
            out = model(minibatch, compute_stress=compute_stress, training=True)
            forces = torch.autograd.grad(
                outputs=out["energy"],  # [B]
                inputs=phi_psi_batch,  # [B, 2]
                grad_outputs=torch.ones_like(out["energy"]),  # [B]
                create_graph=True,
            )[
                0
            ]  # [B, 2]

            # append to the results
            energies_batched += [out["energy"].detach().cpu().numpy().reshape(bs, 1)]
            forces_norm_batched += [
                torch.linalg.norm(forces, dim=1).detach().cpu().numpy().reshape(bs, 1)
            ]
            # forces_normmean_batched += [torch.linalg.norm(forces, dim=1).mean().detach().cpu().numpy().reshape(bs, 1)] # only for more than two dims
            forces_normmean_batched += [
                torch.linalg.norm(forces, dim=1).detach().cpu().numpy().reshape(bs, 1)
            ]
            positions_rotated_batched += [minibatch["positions"].detach().cpu().numpy()]

        # flatten the results
        energies_batched = np.concatenate(energies_batched, axis=0)  # [num_samples, 1]
        energies_batched = energies_batched.reshape(
            len(phi_values), len(psi_values)
        )  # [num_phi, num_psi]
        forces_norm_batched = np.concatenate(
            forces_norm_batched, axis=0
        )  # [num_samples, 1]
        forces_norm_batched = forces_norm_batched.reshape(
            len(phi_values), len(psi_values)
        )  # [num_phi, num_psi]
        forces_normmean_batched = np.concatenate(
            forces_normmean_batched, axis=0
        )  # [num_samples, 1]
        forces_normmean_batched = forces_normmean_batched.reshape(
            len(phi_values), len(psi_values)
        )  # [num_phi, num_psi]
        positions_rotated_batched = np.concatenate(
            positions_rotated_batched, axis=0
        )  # [num_samples, 3]

        # Save the energies to a file
        np.save(datafile, (energies, forces_norm, forces_normmean))
    return energies, forces_norm, forces_normmean


def create_ramachandran_plot(
    phi_range=(-np.pi, np.pi),
    psi_range=(-np.pi, np.pi),
    resolution=360,
    # what to plot
    plot_type="energy",
    show=False,
    show_plt=False,
    convention="andreas",
    # data processing
    log_scale=True,
    recompute=False,
    use_mace=True,
    keep_lowest_energies=-1,  # keep only the lowest keep_lowest_energies% of energies. e.g. 95
    positive_energies=True,
    energy_range=None,
):
    """
    Generate a Ramachandran plot for the alanine dipeptide energy landscape.

    This function samples the φ and ψ dihedral angles over the specified ranges (in radians)
    and computes the corresponding potential energy using the compute_energy_and_forces function.
    A contour plot of the energy (in kJ/mol) is then displayed.

    Parameters:
        show (bool): Whether to display the plot.
        phi_range (tuple): (min_phi, max_phi) in radians (default (-pi, pi)).
        psi_range (tuple): (min_psi, max_psi) in radians (default (-pi, pi)).
        resolution (int): Number of grid points along each dihedral axis (default 60).
        plot_type (str): Type of plot to generate ("energy", "free_energy", "gibbs", "exp").
        convention (str): Convention to use for the dihedral angle indices ("andreas", "bg").
        log_scale (bool): Whether to use a log scale for the energy plot.
        recompute (bool): Whether to recompute the energies or load from file.
        use_mace (bool): Whether to use the MACE model.
        keep_lowest_energies (int): Percentage of highest energies to mask out (default 0).
        positive_energies (bool): Whether to shift all energies to be positive (default True).
        energy_range (tuple): (min_energy, max_energy) in kJ/mol to mask out (default None).

    Returns:
        None. Displays a contour plot.
    """
    print("-" * 80)

    npdtype = np.float128
    torchdtype = torch.float64

    # Create arrays for φ and ψ (in radians)
    phi_values = np.linspace(phi_range[0], phi_range[1], resolution)
    psi_values = np.linspace(psi_range[0], psi_range[1], resolution)

    if use_mace:
        energies, forces_norm, forces_normmean = compute_ramachandran_mace(
            phi_values,
            psi_values,
            recompute=recompute,
            convention=convention,
        )
        unit = "eV"
    else:
        energies, forces_norm, forces_normmean = compute_ramachandran_openmm_amber(
            phi_values,
            psi_values,
            recompute=recompute,
            convention=convention,
        )
        unit = "kJ/mol"

    ############################################################################
    # Convert to the same unit
    ############################################################################
    # http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table.html

    # # Convert energies to kcal/mol
    # # 1 kJ/mol = 0.239006 kcal/mol
    # if unit == "kJ/mol":
    #     energies = energies * 0.239006
    #     unit = "kcal/mol"
    # elif unit == "eV":
    #     # 1 eV = 23.060 980 2 kcal/mol
    #     energies = energies * 23.0609802
    #     unit = "kcal/mol"
    # elif unit == "kcal/mol":
    #     pass
    # else:
    #     raise ValueError(f"Invalid unit: {unit}")

    # Convert energies to kJ/mol
    if unit == "kcal/mol":
        # 1 kcal/mol = 4.18400 kJ/mol
        energies = energies * 4.184
        unit = "kJ/mol"
    elif unit == "eV":
        # 1 eV = 96.486 9 kJ/mol
        energies = energies * 96.4869
        unit = "kJ/mol"
    elif unit == "kJ/mol":
        pass
    else:
        raise ValueError(f"Invalid unit: {unit}")

    ############################################################################
    # Inspect the energies
    ############################################################################
    energies = energies.astype(npdtype)

    # Before printing the five highest energies, add these diagnostic lines:
    print(f"Number of nan values: {np.sum(np.isnan(energies))}")
    print(f"Min energy: {np.nanmin(energies):.1f} [{unit}]")
    print(f"Max energy: {np.nanmax(energies):.1f} [{unit}]")

    # Then modify the existing printing code to ignore nans:
    print(f"The highest energies are:")
    # Get indices of ten highest non-nan values in 2D array
    # print(np.sort(energies.flatten()))
    flat_indices = np.argsort(
        np.where(np.isnan(energies), -np.inf, energies).flatten()
    )[-5:]
    for idx in flat_indices:
        # Convert flat index to 2D indices
        i, j = np.unravel_index(idx, energies.shape)
        print(
            f"  phi={phi_values[i]:6.3f}, psi={psi_values[j]:6.3f} -> {energies[i,j]:8.2f} [{unit}]"
        )
    # The five lowest energies are:
    print(f"The lowest energies are:")
    flat_indices = np.argsort(np.where(np.isnan(energies), np.inf, energies).flatten())[
        :5
    ]
    for idx in flat_indices:
        i, j = np.unravel_index(idx, energies.shape)
        print(
            f"  phi={phi_values[i]:6.3f}, psi={psi_values[j]:6.3f} -> {energies[i,j]:8.2f} [{unit}]"
        )
    # remove the highest energies
    if keep_lowest_energies > 0:
        # example: highest is 120, lowest is 20 -> delta is 100
        # if keep_lowest_energies = 0.9, then we keep the lowest 90%
        # and remove the highest 10%
        print(
            f"Warning: keeping only the lowest {keep_lowest_energies*100:.1f}% of the energy landscape"
        )
        entries_before = np.sum(~np.isnan(energies))
        # delta = np.abs(np.nanmax(energies) - np.nanmin(energies))
        delta = np.nanmax(energies) - np.nanmin(energies)
        print(f" energy range: {delta:.1f} [{unit}]")
        cutoff = np.nanmin(energies) + delta * keep_lowest_energies
        print(f" cutoff: <{cutoff:.1f} [{unit}]")
        energies = np.where(energies < cutoff, energies, np.nan)
        print(
            f" Max energy after removing highest energies: {np.nanmax(energies):.1f} [{unit}]"
        )
        print(
            f" Min energy after removing highest energies: {np.nanmin(energies):.1f} [{unit}]"
        )
        print(f" Entries removed: {entries_before - np.sum(~np.isnan(energies))}")

    if energy_range is not None:
        # boltzmann generator: -128 - -38
        energies = np.where(energies < energy_range[1], energies, np.nan)
        energies = np.where(energies > energy_range[0], energies, np.nan)
        print(f"Elements left in energy range: {np.sum(~np.isnan(energies))}")

    if positive_energies:
        if np.nanmin(energies) <= 0:
            # shift all energies to be positive
            lowest_e = np.nanmin(energies)
            second_lowest_e = np.nanmin(energies[energies > lowest_e])
            _diff = -lowest_e + (second_lowest_e - lowest_e)
            energies += _diff
            print(f"Warning: lowest energy is <=0, adding {_diff:.1f} to all energies")

    ############################################################################
    # Compute free energy / gibbs energy from energies
    ############################################################################
    if unit == "kJ/mol":
        # Energies are in kJ/mol, Forces in kJ/mol/nm
        # kb = 1.380649 * 10^-23 J K^-1
        # kb = 8.314462618 * 10^-3 kJ/(mol⋅K)
        # T = 300 K
        kbT = 0.00831446261815324 * 300.0  # kJ/mol/K
    elif unit == "kcal/mol":
        # Energies are in kcal/mol, Forces in kcal/mol/nm
        # kb = 1.9872041 * 10^-3 kcal/(mol⋅K)
        # kbT = 0.0019872041 * 300.0 # kcal/mol/K
        kbT = 0.0019872041 * 300.0  # kcal/mol/K
    else:
        raise ValueError(f"Invalid unit: {unit}")

    # finite volume element
    dx = ((phi_range[1] - phi_range[0]) / resolution) ** 2

    if plot_type == "free_energy":
        # F(x) = -kBT ln(P(x))
        # P(x) = exp(-E(x)/kbT) / Z
        # Z = sum_x exp(-E(x)/kbT)

        z = np.nansum(np.exp(-energies / kbT))  # * dx
        p = np.exp(-energies / kbT) / z
        # # Use log-sum-exp trick for better numerical stability
        # max_energy = np.max(-energies/kbT)
        # z = np.exp(max_energy) * np.exp(-energies/kbT - max_energy).sum() * dx
        # p = np.exp(-energies/kbT - max_energy) / (z/np.exp(max_energy))

        p += 1e-16
        print(f"p.sum() = {p.sum()} (should be 1)")  # * dx
        free_energies = -kbT * np.log(p)

        # F(x) = -kbT ln[ exp(-E(x)/kbT) / sum_x exp(-E(x)/kbT) ]
        # = -kbT ln[ exp(-E(x)/kbT) ] + kbT ln[ sum_x exp(-E(x)/kbT) ]
        # = E(x) + kbT ln[ sum_x exp(-E(x)/kbT) ]
        free_energies2 = energies + kbT * np.log(z)
        print(
            f"max abs difference between methods: {np.nanmax(np.abs(free_energies - free_energies2))}"
        )

        # free_energies = free_energies2
        if log_scale:
            energies = np.log10(free_energies)
            title = r"$\text{Ramachandran Plot for Alanine Dipeptide: } \log_{10}(F=-k_B T \ln(P))$"
        else:
            energies = free_energies
            title = (
                r"$\text{Ramachandran Plot for Alanine Dipeptide: } F=-k_B T \ln(P)$"
            )

    elif plot_type == "gibbs":
        if log_scale:
            # P(x) = exp(-E(x)/kbT) / Z
            # ln P(x) = -E(x)/kbT - ln Z
            # ln Z = - E(x)/kbT - ln(dx) - ln(sum_x exp(-E(x)/kbT))
            _e = np.where(
                np.isnan(energies), -1.0 * 1e10, energies
            )  # large negative values will become 0 in exp
            ln_p = -energies / kbT - scipy.special.logsumexp(_e / kbT)  # - np.log(dx)
            energies = ln_p
            title = r"$\text{Ramachandran Plot for Alanine Dipeptide: } \log_{10}(P=e^{-U/k_B T})$"
        else:
            z = np.nansum(np.exp(-energies / kbT))  # * dx
            p = np.exp(-energies / kbT) / z
            energies = p
            title = r"$\text{Ramachandran Plot for Alanine Dipeptide: } P=e^{-U/k_B T}$"

    elif plot_type == "exp":
        energies = -energies / kbT
        if log_scale:
            title = r"$\text{Ramachandran Plot for Alanine Dipeptide: } \log_{10}(e^{-U/k_B T})$"
        else:
            print(f"Min arg before exp: {np.nanmin(energies):.1f} [{unit}]")
            print(f"Max arg before exp: {np.nanmax(energies):.1f} [{unit}]")
            energies = np.exp(energies)
            title = r"$\text{Ramachandran Plot for Alanine Dipeptide: } e^{-U/k_B T}$"

    elif plot_type == "energy":
        if log_scale:
            # energies = np.clip(energies, 1e0, None)
            # energies = np.clip(energies, energies.min(), None)
            # energies += np.abs(energies.min()) + 1e0
            # mask out non-positive energies
            energies = np.where(energies > 0, energies, np.nan)
            energies = np.log10(energies)
            title = r"$\text{Ramachandran Plot for Alanine Dipeptide: } \log_{10}(U)$"
        else:
            energies = energies
            title = r"$\text{Ramachandran Plot for Alanine Dipeptide: } U$"

    else:
        raise ValueError(f"Invalid plot type: {plot_type}")

    print(f"Min value in plot: {np.nanmin(energies):.1f} [{unit}]")
    print(f"Max value in plot: {np.nanmax(energies):.1f} [{unit}]")

    ############################################################################
    # Plot the energies, create a contour plot
    ############################################################################

    # values to float32
    energies = energies.astype(np.float64)
    phi_values = phi_values.astype(np.float64)
    psi_values = psi_values.astype(np.float64)

    # plotly has a transposed convention
    energies = energies.T
    forces_norm = forces_norm.T
    forces_normmean = forces_normmean.T
    if log_scale:
        forces_norm = np.log10(forces_norm)
        forces_normmean = np.log10(forces_normmean)

    tempplotfolder = "alanine_dipeptide/plots/"
    tempplotfolder += plot_type
    tempplotfolder += "_mace" if use_mace else "_amber"
    tempplotfolder += "_" + convention
    os.makedirs(tempplotfolder, exist_ok=True)

    # Create a meshgrid for plotting (using 'ij' indexing so that phi_values index the first axis)
    # phi_grid, psi_grid = np.meshgrid(phi_values, psi_values, indexing="ij")

    fig = go.Figure(
        data=go.Contour(
            x=phi_values,
            y=psi_values,
            z=energies,
            colorscale="Viridis",
            type="contour",
            colorbar=dict(
                title="[kJ/mol]",
            ),
            contours=dict(
                start=np.nanmin(energies),
                end=np.nanmax(energies),
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Phi (radians)",
        yaxis_title="Psi (radians)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig_suffix = ""
    if log_scale:
        fig_suffix += "_log"
    if energy_range is not None:
        fig_suffix += f"_range_{energy_range[0]}_{energy_range[1]}"
    if positive_energies:
        fig_suffix += "_positive"
    if keep_lowest_energies > 0:
        fig_suffix += f"_mask{keep_lowest_energies}"
    fig_suffix += ".png"
    figname = f"{tempplotfolder}/ramachandran{fig_suffix}"
    fig.write_image(figname)
    print(f"Saved {figname}")
    if show:
        fig.show()

    ############################################################################
    # plot force norm
    figname = f"{tempplotfolder}/ramachandran_forcenorm"
    title = r"$\text{Norm Force Plot for Alanine Dipeptide } |F|$"
    if log_scale:
        title = r"$\text{Norm Force Plot for Alanine Dipeptide } \log_{10}(|F|)$"
        figname += "_log"
    fig = go.Figure(
        data=go.Contour(
            x=phi_values,
            y=psi_values,
            z=forces_norm,
            colorscale="Viridis",
            type="contour",
            colorbar=dict(
                title=f"[{unit}]",
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Phi (radians)",
        yaxis_title="Psi (radians)",
        margin=dict(l=0, r=5, t=50, b=0),
    )
    figname += ".png"
    fig.write_image(figname)
    print(f"Saved {figname}")

    ############################################################################
    # plot force norm mean
    figname = f"{tempplotfolder}/ramachandran_forcenormmean"
    title = r"$\text{Mean Norm Force Plot for Alanine Dipeptide } \frac{1}{22}\sum_{i=1}^{22}|F_i|$"
    if log_scale:
        title = r"$\text{Mean Norm Force Plot for Alanine Dipeptide } \log_{10}(\frac{1}{22}\sum_{i=1}^{22}|F_i|)$"
        figname += "_log"
    fig = go.Figure(
        data=go.Contour(
            x=phi_values,
            y=psi_values,
            z=forces_normmean,
            colorscale="Viridis",
            type="contour",
            colorbar=dict(
                title=f"[{unit}]",
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Phi (radians)",
        yaxis_title="Psi (radians)",
        margin=dict(l=0, r=5, t=50, b=0),
    )
    figname += ".png"
    fig.write_image(figname)
    print(f"Saved {figname}")


def compare_amber_mace_energies(
    phi_range=(-np.pi, np.pi),
    psi_range=(-np.pi, np.pi),
    resolution=120,
    recompute=False,
    convention="andreas",
):
    """Compare energy landscapes between Amber and MACE models.

    Analyzes:
    1. Energy ranges and statistics
    2. Correlation between high-energy regions
    3. Location of minima/maxima
    """
    print("\nComparing Amber vs MACE energy landscapes:")
    print("-" * 80)

    # Create arrays for φ and ψ (in radians)
    phi_values = np.linspace(phi_range[0], phi_range[1], resolution)
    psi_values = np.linspace(psi_range[0], psi_range[1], resolution)
    phi_psi_values = np.array(np.meshgrid(phi_values, psi_values)).T.reshape(-1, 2)

    # Get energies from both models
    amber_energies, _, _ = compute_ramachandran_openmm_amber(
        phi_values=phi_values,
        psi_values=psi_values,
        convention=convention,
    )
    # units are kJ/mol

    mace_energies, _, _ = compute_ramachandran_mace(
        phi_values=phi_values,
        psi_values=psi_values,
        convention=convention,
    )
    # Mace units are eV
    # 1 eV = 96.486 9 kJ/mol
    mace_energies = mace_energies * 96.4869
    unit = "kJ/mol"

    # Compare energy ranges
    print("\nEnergy ranges (kJ/mol):")
    print(
        f"Amber: [{np.nanmin(amber_energies):.2f}, {np.nanmax(amber_energies):.2f}],",
        f"range={np.abs(np.nanmin(amber_energies) - np.nanmax(amber_energies)):.2f} [{unit}]",
    )
    print(
        f"MACE:  [{np.nanmin(mace_energies):.2f}, {np.nanmax(mace_energies):.2f}],",
        f"range={np.abs(np.nanmin(mace_energies) - np.nanmax(mace_energies)):.2f} [{unit}]",
    )

    print("\nEnergy statistics (kJ/mol):")
    print(
        f"Amber mean: {np.nanmean(amber_energies):.2f}, std: {np.nanstd(amber_energies):.2f}"
    )
    print(
        f"MACE mean:  {np.nanmean(mace_energies):.2f}, std: {np.nanstd(mace_energies):.2f}"
    )

    # Find locations of highest energies
    n_highest = 5
    amber_highest_idx = np.argpartition(amber_energies.flatten(), -n_highest)[
        -n_highest:
    ]
    mace_highest_idx = np.argpartition(mace_energies.flatten(), -n_highest)[-n_highest:]

    # sort energies in descending order and corresponding phi, psi values
    amber_highest_energy_idx = np.argsort(amber_energies.flatten())[::-1]
    mace_highest_energy_idx = np.argsort(mace_energies.flatten())[::-1]
    # list of [phi, psi] in the same order
    amber_highest_phi_psi = phi_psi_values[amber_highest_energy_idx]
    mace_highest_phi_psi = phi_psi_values[mace_highest_energy_idx]

    print(f"\nTop {n_highest} highest energy configurations:")
    print("\nAmber highest energy locations (φ,ψ) and corresponding MACE energies:")
    for idx in amber_highest_idx:
        i, j = np.unravel_index(idx, amber_energies.shape)
        phi, psi = phi_values[i], psi_values[j]
        # find the same phi_psi in mace_highest_phi_psi
        mace_ranking = np.where(np.all(mace_highest_phi_psi == [phi, psi], axis=1))[0][
            0
        ]
        print(
            f"φ={phi:.2f}, ψ={psi:.2f}: Amber={amber_energies[i,j]:.2f}, MACE={mace_energies[i,j]:.2f} (rank {mace_ranking})"
        )

    print("\nMACE highest energy locations (φ,ψ) and corresponding Amber energies:")
    for idx in mace_highest_idx:
        i, j = np.unravel_index(idx, mace_energies.shape)
        phi, psi = phi_values[i], psi_values[j]
        # find the same phi_psi in amber_highest_phi_psi
        amber_ranking = np.where(np.all(amber_highest_phi_psi == [phi, psi], axis=1))[
            0
        ][0]
        print(
            f"φ={phi:.2f}, ψ={psi:.2f}: MACE={mace_energies[i,j]:.2f}, Amber={amber_energies[i,j]:.2f} (rank {amber_ranking})"
        )

    # Calculate correlation coefficient
    # valid_mask = ~np.isnan(amber_energies) & ~np.isnan(mace_energies)
    correlation = np.corrcoef(amber_energies.flatten(), mace_energies.flatten())[0, 1]
    print(f"\nCorrelation coefficient between energies: {correlation:.3f}")

    # Calculate the correlation between the phi, psi values
    correlation = np.corrcoef(
        amber_highest_phi_psi.flatten(), mace_highest_phi_psi.flatten()
    )[0, 1]
    print(
        f"\nCorrelation coefficient between phi, psi values sorted by energy: {correlation:.3f}"
    )


############################################################################
# Main
############################################################################
if __name__ == "__main__":

    # Amber force field
    # create_ramachandran_plot(
    #     use_mace=False,
    #     log_scale=False,
    #     positive_energies=True,
    #     plot_type="exp",
    # )
    # create_ramachandran_plot(
    #     use_mace=False,
    #     log_scale=False,
    #     plot_type="gibbs",
    # )

    # create_ramachandran_plot(
    #     use_mace=False,
    #     log_scale=False,
    #     positive_energies=False,
    #     energy_range=(-128, -38),
    #     plot_type="energy",
    #     convention="andreas",
    # )
    # create_ramachandran_plot(
    #     use_mace=False,
    #     log_scale=True,
    #     positive_energies=True,
    #     # energy_range=(-128, -38),
    #     plot_type="energy",
    #     convention="andreas",
    # )
    # create_ramachandran_plot(
    #     use_mace=False,
    #     log_scale=False,
    #     positive_energies=False,
    #     energy_range=(-128, -38),
    #     plot_type="energy",
    #     convention="bg",
    # )
    # create_ramachandran_plot(
    #     use_mace=False,
    #     log_scale=True,
    #     positive_energies=True,
    #     # energy_range=(-128, -38),
    #     plot_type="energy",
    #     convention="bg",
    # )

    # create_ramachandran_plot(
    #     recompute=False,
    #     use_mace=False,
    #     log_scale=False,
    #     show_plt=True,
    #     positive_energies=True,
    #     plot_type="exp",
    # )

    ############################################################
    # MACE force field
    ############################################################
    create_ramachandran_plot(
        use_mace=True,
        log_scale=False,
        plot_type="energy",
        positive_energies=True,
    )
    create_ramachandran_plot(
        use_mace=True,
        log_scale=True,
        show_plt=True,
        plot_type="energy",
        positive_energies=True,
    )
    create_ramachandran_plot(
        use_mace=True,
        log_scale=False,
        plot_type="gibbs",
        positive_energies=True,
    )

    # Our loss is the unnormalized Boltzmann distribution = exp(-E/kbT)
    # order: keep_lowest_energies, energy_range, positive_energies
    create_ramachandran_plot(
        use_mace=True,
        log_scale=False,
        plot_type="exp",
        positive_energies=True,
    )
    # log scale
    create_ramachandran_plot(
        use_mace=True,
        log_scale=True,
        plot_type="exp",
        positive_energies=True,
    )
    # mask out highest 10%
    create_ramachandran_plot(
        use_mace=True,
        log_scale=False,
        plot_type="exp",
        positive_energies=True,
        keep_lowest_energies=0.5,
    )
    # mask out highest 10%, log scale
    create_ramachandran_plot(
        use_mace=True,
        log_scale=True,
        plot_type="exp",
        positive_energies=True,
        keep_lowest_energies=0.5,
    )

    ############################################################
    # Compare Amber and MACE energy landscapes
    ############################################################
    compare_amber_mace_energies()
