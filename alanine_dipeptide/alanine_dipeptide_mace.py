import numpy as np
import torch
import os
import pathlib
import math
from tqdm import tqdm

import mace
from mace.calculators import mace_off, mace_anicc
from mace.tools import torch_geometric, torch_tools, utils
import mace.data
import ase
import ase.io
import ase.build
from ase.calculators.calculator import Calculator, all_changes
import openmm
from openmm.unit import nanometer
from openmm.unit import Quantity

from dihedral import (
    set_dihedral_torch,
    set_dihedral_torch_batched,
    compute_dihedral_torch,
    compute_dihedral_torch_batched,
    update_neighborhood_graph_batched,
)
from mace_neighbourshood import get_neighborhood
from alanine_dipeptide_openmm_amber99 import fffile, pdbfile

import torch_geometric as tg


"""Alanine dipeptide with using dihedral angles as reaction coordinates.

Transforms dihedral angles to atom positions.
"""


# def build_alanine_dipeptide_openmm(phi, psi, pdbfile):
#     """
#     Build a full-atom 3D configuration of alanine dipeptide with the specified
#     backbone dihedral angles phi and psi (in radians).

#     The function loads a template PDB file and then
#     rotates the appropriate groups of atoms so that the dihedrals match the desired values.

#     Parameters:
#       phi: Desired φ dihedral angle (radians)
#       psi: Desired ψ dihedral angle (radians)

#     Returns:
#       positions_quantity: A simtk.unit.Quantity (shape: [N_atoms, 3]) in nanometers.

#     > **Note:** Adjust the following dihedral atom indices to match your template.
#     """
#     # Load the template structure.
#     pdb = openmm.app.PDBFile(pdbfile)
#     # Extract positions as a NumPy array in nanometers.
#     positions = torch.tensor(pdb.positions.value_in_unit(nanometer))

#     positions = set_dihedral_torch(positions, "phi", phi, "phi")
#     positions = set_dihedral_torch(positions, "psi", psi, "psi")

#     return positions


def load_alanine_dipeptide_ase():
    """
    Build a full-atom 3D configuration of alanine dipeptide using ASE.

    Parameters:
      phi: Desired φ dihedral angle (radians)
      psi: Desired ψ dihedral angle (radians)
      pdbfile: Path to PDB template file

    Returns:
      atoms: ASE Atoms object with updated positions
    """
    # Load PDB into ASE
    atoms = ase.io.read(pdbfile)
    return fix_atomic_numbers(atoms)


# from mace.calculators.mace import MACECalculator
def _atoms_to_batch(calc: mace.calculators.mace.MACECalculator, atoms, bs=1, repeats=1):
    config = mace.data.config_from_atoms(atoms, charges_key=calc.charges_key)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            mace.data.AtomicData.from_config(
                config, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.heads
            )
        ]
        * repeats,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to(calc.device)
    return batch


def update_alanine_dipeptide_ase(
    atoms, phi_psi: torch.Tensor = None, convention="andreas"
):
    """
    Update the positions of alanine dipeptide with the specified
    backbone dihedral angles phi and psi (in radians) using ASE.

    Parameters:
      phi_psi: Desired φ and ψ dihedral angles (radians)
      pdbfile: Path to PDB template file

    Returns:
      atoms: ASE Atoms object with updated positions
    """
    # Get positions as torch tensor
    positions = torch.tensor(atoms.get_positions())

    # Set dihedral angles
    positions = set_dihedral_torch(positions, "phi", phi_psi[0], "phi", convention)
    positions = set_dihedral_torch(positions, "psi", phi_psi[1], "psi", convention)

    # Update positions in ASE Atoms object
    atoms.set_positions(positions.numpy())

    return atoms


# DEPRECATED: use update_alanine_dipeptide_with_grad_batched instead
def update_alanine_dipeptide_with_grad(
    phi_psi: torch.Tensor, batch: dict, set_phi=True, set_psi=True, convention="andreas"
) -> dict:
    """
    Update positions based on phi_psi angles while maintaining gradient flow.
    Works with MACE Batch objects.
    """
    # Create a new batch with the same attributes as the input batch
    if not isinstance(batch, dict):
        batch = batch.to_dict()
    new_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    # Get positions tensor and apply dihedral rotations
    positions = batch["positions"].clone()

    if set_phi:
        positions = set_dihedral_torch(positions, "phi", phi_psi[0], "phi", convention)
    if set_psi:
        positions = set_dihedral_torch(positions, "psi", phi_psi[1], "psi", convention)

    # Update positions in the new batch
    new_batch["positions"] = positions

    return new_batch


def update_alanine_dipeptide_with_grad_batched(
    phi_psi: torch.Tensor,
    batch: dict,
    set_phi=True,
    set_psi=True,
    convention="andreas",
    return_unbatched=False,
) -> dict:
    """
    Update positions based on phi_psi angles while maintaining gradient flow.
    Works with MACE Batch objects.
    phi_psi is a tensor of shape (B, 2) where each row is [phi, psi], or just (2) for a single configuration.
    """
    # Create a new batch with the same attributes as the input batch
    if not isinstance(batch, dict):
        batch = batch.to_dict()
    new_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    if phi_psi.dim() <= 1:
        phi_psi = phi_psi.unsqueeze(0)

    # Get positions tensor and apply dihedral rotations
    positions = batch["positions"].clone()

    bs = batch["batch"].max() + 1

    # reshape positions from (B*N_atoms, 3) to (B, N_atoms, 3) based on batch["batch"]
    positions = tg.utils.unbatch(src=batch["positions"], batch=batch["batch"], dim=0)
    positions = torch.stack(list(positions), dim=0).reshape(bs, -1, 3)

    if set_phi:
        positions = set_dihedral_torch_batched(
            positions, "phi", phi_psi[:, 0], "phi", convention
        )
    if set_psi:
        positions = set_dihedral_torch_batched(
            positions, "psi", phi_psi[:, 1], "psi", convention
        )

    # Update positions in the new batch
    if return_unbatched:
        new_batch["positions"] = positions
    else:
        new_batch["positions"] = positions.reshape(-1, 3)

    return new_batch


def fix_atomic_numbers(atoms):
    """
    ASE is misinterpreting the "CA" label from your PDB file.
    atom index 8 is showing up as "Ca" (Calcium, atomic number 20) when it should be "CA" (a Carbon atom).
    In PDB format, "CA" is a special atom name that stands for the alpha Carbon (Cα) of an amino acid, but ASE is interpreting it as the chemical symbol for Calcium.
    Fix atomic numbers in the ASE Atoms object to match what MACE expects.
    For alanine dipeptide, we should only have H(1), C(6), N(7), and O(8).
    """
    atomic_numbers = atoms.get_atomic_numbers()
    # Find any calcium atoms (atomic number 20) and convert them to carbon (atomic number 6)
    atomic_numbers[atomic_numbers == 20] = 6
    atoms.set_atomic_numbers(atomic_numbers)
    return atoms


def compute_energy_and_forces_mace(
    dihedrals: torch.Tensor,
    batch_base: dict = None,
    model: torch.nn.Module = None,
    model_type: str = "off",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the energy and forces for a given set of dihedral angles using the MACE model.
    """
    if batch_base is None or model is None:
        # get alanine dipeptide atoms
        atoms = load_alanine_dipeptide_ase()

        # Get MACE force field: mace_off or mace_anicc
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        if model_type == "off":
            calc = mace_off(model="medium", device=device_str, enable_cueq=True)
        elif model_type == "anicc":
            calc = mace_anicc(device=device_str, enable_cueq=True)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        device = calc.device
        atoms.calc = calc
        atoms_calc = atoms.calc

        ################################################
        # ASE atoms -> torch batch: one time setup
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

    # Create phi_psi tensor with gradients
    phi_psi = dihedrals
    phi_psi.requires_grad = True

    # Update positions
    batch = update_alanine_dipeptide_with_grad(phi_psi, batch_base)

    # need to update edge_index
    # https://github.com/ACEsuit/mace/blob/3e578b02e649a5b2ac8109fa857698fdc42cf842/mace/modules/models.py#L72
    # no gradients for these, but should not affect forces
    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        positions=batch["positions"].detach().cpu().numpy(),
        cutoff=model.r_max.item(),
        cell=batch["cell"].detach().cpu().numpy(),
    )
    batch["edge_index"] = torch.tensor(
        edge_index, device=device, dtype=batch["edge_index"].dtype
    )
    batch["shifts"] = torch.tensor(shifts, device=device, dtype=batch["shifts"].dtype)
    batch["unit_shifts"] = torch.tensor(
        unit_shifts, device=device, dtype=batch["unit_shifts"].dtype
    )
    batch["cell"] = torch.tensor(cell, device=device, dtype=batch["cell"].dtype)

    # Compute energy by calling MACE
    out = model(
        batch,
        compute_stress=compute_stress,
        # training=True -> retain_graph when calculating forces=dE/dx
        # which is what we need to compute forces'=dE/dphi_psi
        training=True,  # atoms_calc.use_compile,
    )

    # Compute forces
    forces = torch.autograd.grad(out["energy"], phi_psi, create_graph=True)[0]

    return out["energy"], forces


############################################################################
# Test functions
############################################################################


def test_mace_alanine_dipeptide():
    print("-" * 80)
    atoms = load_alanine_dipeptide_ase()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Print out the atoms list
    print("\nList of atoms:")
    print("Index  Symbol  Atomic#  Position (x, y, z)")
    print("-" * 45)
    for i, (sym, num, pos) in enumerate(
        zip(
            atoms.get_chemical_symbols(),
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
        )
    ):
        print(
            f"{i:3d}     {sym:2s}      {num:2d}      ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})"
        )

    # Get MACE force field
    calc = mace_off(model="medium", device=device_str, enable_cueq=False)
    device = calc.device
    atoms.calc = calc

    # Test energy
    print("Energy:", atoms.get_potential_energy())

    # Update positions
    atoms = update_alanine_dipeptide_ase(atoms, phi_psi=torch.tensor([0.0, 0.0]))
    print("Energy:", atoms.get_potential_energy())
    # print("Forces:", atoms.get_forces())
    return True


def test_mace_alanine_dipeptide_dihedral_grad():
    # compute force w.r.t. phi and psi
    print("-" * 80)

    # get alanine dipeptide atoms
    atoms = load_alanine_dipeptide_ase()

    # Get MACE force field: mace_off or mace_anicc
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_off(model="medium", device=device_str, enable_cueq=False)
    device = calc.device
    atoms.calc = calc
    atoms_calc = atoms.calc

    ################################################
    # ASE atoms -> torch batch: one time setup
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

    # save default configuration
    ase.io.write(f"alanine_dipeptide/outputs/default.xyz", atoms)

    # phi_indices = [6, 8]  # C-N-CA-C for phi
    # psi_indices = [6, 8, 14, 16]  # N-CA-C-N for psi
    phi_indices = [1, 2, 3, 4]  # Adjust indices to match template
    psi_indices = [2, 3, 4, 5]

    # only plot atoms in phi_indices
    _atoms = atoms.copy()
    for _a in _atoms:
        if _a.index not in phi_indices:
            _a.symbol = "He"
    ase.io.write(f"alanine_dipeptide/outputs/default_phi_atoms.xyz", _atoms)
    # only plot atoms in psi_indices
    _atoms = atoms.copy()
    for _a in _atoms:
        if _a.index not in psi_indices:
            _a.symbol = "He"
    ase.io.write(f"alanine_dipeptide/outputs/default_psi_atoms.xyz", _atoms)

    # set phi and psi to 0
    phi_psi = torch.tensor([0.0, 0.0], requires_grad=True)
    batch = update_alanine_dipeptide_with_grad(
        phi_psi, batch_base, set_phi=True, set_psi=False
    )
    _atoms = atoms.copy()
    _atoms.set_positions(batch["positions"].detach().cpu().numpy())
    ase.io.write(f"alanine_dipeptide/outputs/default_phi0.xyz", _atoms)
    batch = update_alanine_dipeptide_with_grad(
        phi_psi, batch_base, set_phi=False, set_psi=True
    )
    _atoms = atoms.copy()
    _atoms.set_positions(batch["positions"].detach().cpu().numpy())
    ase.io.write(f"alanine_dipeptide/outputs/default_psi0.xyz", _atoms)

    ################################################
    # placeholder loop for training or plotting
    angles = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    for i, phi_psi in enumerate(angles):
        # Create phi_psi tensor with gradients
        phi_psi = torch.tensor(phi_psi, requires_grad=True)

        # Update positions
        batch = update_alanine_dipeptide_with_grad(phi_psi, batch_base)

        # need to update edge_index
        # https://github.com/ACEsuit/mace/blob/3e578b02e649a5b2ac8109fa857698fdc42cf842/mace/modules/models.py#L72
        # no gradients for these, but should not affect forces
        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=batch["positions"].detach().cpu().numpy(),
            cutoff=model.r_max.item(),
            cell=batch["cell"].detach().cpu().numpy(),
        )
        batch["edge_index"] = torch.tensor(
            edge_index, device=device, dtype=batch["edge_index"].dtype
        )
        batch["shifts"] = torch.tensor(
            shifts, device=device, dtype=batch["shifts"].dtype
        )
        batch["unit_shifts"] = torch.tensor(
            unit_shifts, device=device, dtype=batch["unit_shifts"].dtype
        )
        batch["cell"] = torch.tensor(cell, device=device, dtype=batch["cell"].dtype)

        # Test gradient flow
        # testgrad = torch.autograd.grad(batch["positions"].sum(), phi_psi, create_graph=True)[0]
        # print("Test gradient w.r.t. phi_psi:", testgrad)

        # Compute energy by calling MACE
        out = model(
            batch,
            compute_stress=compute_stress,
            # training=True -> retain_graph when calculating forces=dE/dx
            # which is what we need to compute forces'=dE/dphi_psi
            training=True,  # atoms_calc.use_compile,
        )

        # Compute forces
        forces = torch.autograd.grad(out["energy"], phi_psi, create_graph=True)[0]
        print(f"Forces w.r.t. phi_psi: {forces}")

        # save as .xyz file
        atoms.set_positions(batch["positions"].detach().cpu().numpy())
        ase.io.write(
            f"alanine_dipeptide/outputs/phi{phi_psi[0]:.1f}_psi{phi_psi[1]:.1f}.xyz",
            atoms,
        )


def test_mace_alanine_dipeptide_batching():
    print("-" * 60)
    # Create a batch of dihedral angle pairs (in radians). For example, B=3.
    # Each row: [phi, psi]
    dihedrals_batch = torch.tensor(
        [
            [-60 * np.pi / 180, -45 * np.pi / 180],
            [-80 * np.pi / 180, 30 * np.pi / 180],
            [50 * np.pi / 180, 60 * np.pi / 180],
            [0 * np.pi / 180, 0 * np.pi / 180],
            [30 * np.pi / 180, 0 * np.pi / 180],
            [30 * np.pi / 180, 30 * np.pi / 180],
            [5 * np.pi / 180, 180 * np.pi / 180],
        ],
        dtype=torch.float32,
    )

    energies_list = []
    forces_list = []

    # Compute energies and forces for each configuration sequentially
    for i in range(dihedrals_batch.shape[0]):
        print(
            f"Configuration {i}: φ = {dihedrals_batch[i,0]*180/np.pi:.1f}°, ψ = {dihedrals_batch[i,1]*180/np.pi:.1f}°"
        )
        energy, forces = compute_energy_and_forces_mace(
            dihedrals_batch[i], model_type="off"
        )
        # 1 kcal/mol = 0.04336 eV
        # 1 eV = 23.0605 kcal/mol
        print(
            f"  Energy (eV): {energy.item():.2f} = {energy.item()*23.0605:.2f} kcal/mol"
        )
        energies_list.append(energy.item())
        forces_list.append(forces)

    min_energy = min(energies_list)
    max_energy = max(energies_list)

    print(
        f"Energy range: {(max_energy - min_energy):.2f} eV = {(max_energy - min_energy)*23.0605:.2f} kcal/mol"
    )

    # compute in parallel
    raise NotImplementedError("compute_energy_and_forces_mace_batched not implemented")
    energies_parallel, forces_parallel = compute_energy_and_forces_mace(
        dihedrals_batch, model_type="off"
    )

    # check if parallel and sequential energies and forces are the same
    assert torch.allclose(energies, energies_parallel)
    assert torch.allclose(forces, forces_parallel)


# Most important test here
def test_mace_ramachandran_batched_vs_non_batched(
    phi_range=(-np.pi, np.pi),
    psi_range=(-np.pi, np.pi),
    resolution=4,
    convention="andreas",
    dtypestr="float64",
):
    """
    Compute the Ramachandran plot for the alanine dipeptide energy landscape using the MACE model.
    Takes forces as derivative w.r.t. dihedral angles.
    """
    print("-" * 60)
    # Create arrays for φ and ψ (in radians)
    phi_values = np.linspace(phi_range[0], phi_range[1], resolution)
    psi_values = np.linspace(psi_range[0], psi_range[1], resolution)
    # if file exists, load it
    # datafile = f"alanine_dipeptide/outputs/ramachandran_mace_{resolution}_{convention}.npy"
    # recompute = os.path.exists(datafile) and not recompute
    recompute = True
    if not recompute:
        energies, forces_norm, forces_normmean = np.load(datafile)
    else:
        # get alanine dipeptide atoms
        atoms = load_alanine_dipeptide_ase()

        # Get MACE force field: mace_off or mace_anicc
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_off(model="medium", device=device_str, dtype=dtypestr, enable_cueq=True)
        # calc = mace_anicc(device=device_str, dtype=dtypestr, enable_cueq=True)
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

        # Initialize an array to store energies.
        positions_rotated = np.zeros(
            (len(phi_values) * len(psi_values) * batch_base["positions"].shape[0], 3)
        )
        energies = np.zeros((len(phi_values), len(psi_values)))
        forces_norm = np.zeros((len(phi_values), len(psi_values)))
        forces_normmean = np.zeros((len(phi_values), len(psi_values)))

        n_atoms = batch_base["positions"].shape[0]

        # Loop over grid points.
        for i, phi in tqdm(enumerate(phi_values), total=len(phi_values)):
            for j, psi in enumerate(psi_values):
                # Angles already in radians
                dihedrals = torch.tensor([phi, psi], dtype=torch.float32)
                dihedrals.requires_grad = True

                # Update positions
                batch = update_alanine_dipeptide_with_grad(
                    dihedrals, batch_base, convention
                )

                # need to update edge_index
                # no gradients for these, but should not affect forces
                edge_index, shifts, unit_shifts, cell = get_neighborhood(
                    positions=batch["positions"].detach().cpu().numpy(),
                    cutoff=model.r_max.item(),
                    cell=batch["cell"].detach().cpu().numpy(),
                )
                batch["edge_index"] = torch.tensor(
                    edge_index, device=device, dtype=batch["edge_index"].dtype
                )
                batch["shifts"] = torch.tensor(
                    shifts, device=device, dtype=batch["shifts"].dtype
                )
                batch["unit_shifts"] = torch.tensor(
                    unit_shifts, device=device, dtype=batch["unit_shifts"].dtype
                )
                batch["cell"] = torch.tensor(
                    cell, device=device, dtype=batch["cell"].dtype
                )
                # TODO: why is the cell so huge?
                # print(f"\n cell after: {batch['cell']}")
                # print(f" cell before: {batch_base['cell']}")
                positions_rotated[i * n_atoms : (i + 1) * n_atoms] = (
                    batch["positions"].detach().cpu().numpy()
                )

                # Compute energy by calling MACE
                out = model(
                    batch,
                    compute_stress=compute_stress,
                    # training=True -> retain_graph when calculating forces=dE/dx
                    # which is what we need to compute forces'=dE/dphi_psi
                    training=True,  # atoms_calc.use_compile,
                )

                # Compute forces
                forces = torch.autograd.grad(
                    out["energy"], dihedrals, create_graph=True
                )
                if isinstance(forces, tuple):
                    forces = forces[0]

                # Store the energy (in kJ/mol)
                energy = out["energy"]
                energies[i, j] = energy.item()
                # force are shape [22,3] each row is a force vector for an atom
                forces_norm[i, j] = torch.linalg.norm(forces).item()
                forces_normmean[i, j] = torch.linalg.norm(forces, axis=-1).mean().item()

                # print the computed energy for each grid point.
                # tqdm.write(
                #     f"phi={phi:6.3f}, psi={psi:6.3f} -> U={energy.item():8.2f} eV?, F={forces_norm[i, j]:8.2f}"
                # )
        ############################################################################
        # now do it again with batched variants and compare
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
        batch_size = 9
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

            # np.abs(minibatch["positions"][:22].detach().cpu().numpy() - positions_rotated[:22])
            # np.abs(minibatch["positions"][:22].detach().cpu().numpy() - positions_rotated_batched[i*n_atoms*batch_size:(i+1)*n_atoms*batch_size])

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

        # Compare results
        energy_diff = np.abs(energies - energies_batched)
        forces_diff = np.abs(forces_norm - forces_norm_batched)
        print(
            f"Mean rel difference in energies: {energy_diff.mean()/energies.mean():.6f} (max_rel={energy_diff.max()/energies.max():.6f})"
        )
        print(
            f"Mean rel difference in forces: {forces_diff.mean()/forces_norm.mean():.6f} (max_rel={forces_diff.max()/forces_norm.max():.6f})"
        )

        # Save the energies to a file
        # np.save(datafile, (energies, forces_norm, forces_normmean))
    return energies, forces_norm, forces_normmean


if __name__ == "__main__":
    test_mace_alanine_dipeptide()
    test_mace_alanine_dipeptide_dihedral_grad()
    # test_mace_alanine_dipeptide_batching()
    test_mace_ramachandran_batched_vs_non_batched()
