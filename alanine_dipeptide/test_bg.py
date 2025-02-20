import os
import numpy as np
import torch

import openmm
import openmm.app as app
import openmm.unit as unit

from bg import *
import bgmol

import copy

import mdtraj as md

from alanine_dipeptide_openmm_amber99 import pdbfile


def test_ic_transform_energy_consistency():
    print("-" * 80)
    # -------------------------
    # Load the dataset and OpenMM system
    # -------------------------
    # Downloading http://ftp.mi.fu-berlin.de/pub/cmb-data/bgmol/datasets/ala2/Ala2Implicit300.tgz to bgflow/Ala2Implicit300.tgz
    is_data_here = os.path.isfile("Ala2TSF300.tgz")
    dataset = Ala2Implicit300(download=not is_data_here, read=True)
    alanine_system = dataset.system
    temperature = dataset.temperature

    # Set up a simple OpenMM simulation
    integrator = openmm.LangevinIntegrator(temperature, 1, 0.001)
    simulation = app.Simulation(
        alanine_system.topology, alanine_system.system, integrator
    )

    # ----------------------------------------------------------
    # Setup a second simulation with the pdb file
    # ----------------------------------------------------------

    pdb = openmm.app.PDBFile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "alanine-dipeptide-nowater.pdb",
        )
    )
    ff = openmm.app.ForceField("amber99sbildn.xml", "amber96_obc.xml")
    _system = ff.createSystem(
        pdb.getTopology(),
        removeCMMotion=True,
        nonbondedMethod=openmm.app.NoCutoff,
        constraints=openmm.app.HBonds,
        rigidWater=True,
    )
    # _system.reinitialize_energy_model(temperature=temperature)
    _integrator = openmm.LangevinIntegrator(temperature, 1, 0.001)
    # _simulation = app.Simulation(pdb.getTopology(), _system, _integrator)
    _simulation = app.Simulation(pdb.getTopology(), _system, _integrator)

    # system2 = bgmol.systems.ala2.AlanineDipeptideImplicit()
    # print(f"Energy difference pdb: {np.abs(energy_pdb - energy_dataset):.3f} kJ/mol")

    system_bgmol = system_by_name("AlanineDipeptideImplicit")
    integrator_bgmol = openmm.LangevinIntegrator(temperature, 1, 0.001)
    simulation_bgmol = app.Simulation(
        system_bgmol.topology, system_bgmol.system, integrator_bgmol
    )

    # ----------------------------------------------------------
    # Compare energies in dataset to calculated energies to pdb
    # ----------------------------------------------------------

    # Pick 3 random frames from the trajectory
    n_frames = len(dataset.trajectory)
    random_indices = np.random.choice(n_frames, size=5, replace=False)
    print("\nComparing  energies for random frames:")
    print("-------------------------------------------------------")

    for idx in random_indices:
        # Get positions for this frame from the trajectory
        positions = dataset.trajectory[idx].xyz[0]  # Get numpy array of positions

        # Calculated with system from dataset
        simulation.context.setPositions(copy.deepcopy(positions))
        state = simulation.context.getState(getEnergy=True)
        energy_calculated = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

        # Calculated with system from pdb
        _simulation.context.setPositions(copy.deepcopy(positions))
        _state = _simulation.context.getState(getEnergy=True)
        energy_calculated_pdb = _state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

        # Calculated with system from bgmol
        simulation_bgmol.context.setPositions(copy.deepcopy(positions))
        state_bgmol = simulation_bgmol.context.getState(getEnergy=True)
        energy_calculated_bgmol = state_bgmol.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

        # Get reference energy from dataset
        energy_dataset = dataset.energies[idx]

        print(f"\nFrame {idx}:")
        # print(f"Position reconstruction error: {np.linalg.norm(positions - positions_recon_np):.6f}")
        print(
            f"Energy difference: {np.abs(energy_calculated - energy_dataset):.3f} kJ/mol"
        )
        print(
            f"Energy difference pdb: {np.abs(energy_calculated_pdb - energy_calculated):.3f} kJ/mol ({energy_calculated:.1f})"
        )
        print(
            f"Energy difference bgmol: {np.abs(energy_calculated_bgmol - energy_calculated):.3f} kJ/mol ({energy_calculated:.1f})"
        )


def test_system_traj_same_as_temp_traj(phi_indices, psi_indices):
    print("-" * 80)
    is_data_here = os.path.isdir("Ala2TSF300")
    dataset = Ala2Implicit300(download=not is_data_here, read=True)

    # The dataset contains forces, energies and coordinates it also holds a reference to the system that defines the potential energy function.

    openmmsystem = dataset.system
    system = dataset.system

    positions = dataset.trajectory.xyz[0]

    # The system is an OpenMMSystem object, it provides access to the openmm.system instance,
    # the topology, and a set of initial coordinates.

    integrator = LangevinIntegrator(dataset.temperature, 1, 0.001)
    simulation = openmm.app.Simulation(
        openmmsystem.topology, openmmsystem.system, integrator
    )
    simulation.context.setPositions(positions)

    # V1: from dataset with system.compute_phi_psi
    phi, psi = dataset.system.compute_phi_psi(dataset.trajectory)
    print("phi system.compute_phi_psi", phi)
    print("psi system.compute_phi_psi", psi)

    # V1.5
    phi_mdtraj = md.compute_dihedrals(dataset.trajectory, [phi_indices])[0][0]
    psi_mdtraj = md.compute_dihedrals(dataset.trajectory, [psi_indices])[0][0]
    print("phi compute_dihedral", phi_mdtraj)
    print("psi compute_dihedral", psi_mdtraj)

    # V2: from pdb file
    traj = md.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "alanine-dipeptide-nowater.pdb",
        )
    )
    traj.xyz[0] = positions
    phi_pdb = md.compute_dihedrals(traj, [phi_indices])[0][0]
    psi_pdb = md.compute_dihedrals(traj, [psi_indices])[0][0]
    print("phi pdb", phi_pdb)
    print("psi pdb", psi_pdb)

    # V3: temporary traj object from pdb
    pdb = openmm.app.PDBFile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "alanine-dipeptide-nowater.pdb",
        )
    )
    traj = md.Trajectory(xyz=positions, topology=pdb.topology)
    phi_traj = md.compute_dihedrals(traj, [phi_indices])[0][0]
    psi_traj = md.compute_dihedrals(traj, [psi_indices])[0][0]
    print("phi temp traj pdb topology", phi_traj)
    print("psi temp traj pdb topology", psi_traj)


if __name__ == "__main__":
    from dihedral import phi_indices, psi_indices, phi_atoms_bg, psi_atoms_bg

    test_ic_transform_energy_consistency()

    test_system_traj_same_as_temp_traj(phi_indices, psi_indices)
