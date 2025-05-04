#!/usr/bin/env python
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=plot
#SBATCH --output=plot.out
#SBATCH --error=plot.out

import argparse
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt

# Default geometry file
AMP_GEOMS = "amp_qmmm_geoms.extxyz"

# Keywords for extracting data
REF_ENERGY_KEY = "qm_energies_ref"
REF_FORCES_KEY = "qm_gradients_ref"
REF_DIPOLE_KEY = "dipole_ref"
REF_QUADRUPOLE_KEY = "quadrupole_ref"
PRED_ENERGY_KEY = "qm_energies_pred"
PRED_FORCES_KEY = "qm_gradients_pred"
PRED_DIPOLE_KEY = "dipole_pred"
PRED_QUADRUPOLE_KEY = "quadrupole_pred"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plotting script for AMP")
    parser.add_argument("-g", "--geoms", type=str, default=AMP_GEOMS, \
                        help="Path to a geometry file containing both reference and predicted data. Run test_amp.py to generate this file.")
    return parser.parse_args()

def get_ref(mols, energy_keyword=None, forces_keyword=None, dipole_keyword=None, quadrupole_keyword=None):
    ref_energy = []
    ref_forces = []
    ref_dipoles = []
    ref_quadrupoles = []
    for m in mols:
        if dipole_keyword and dipole_keyword in m.info:
            ref_dipoles.extend(m.info[dipole_keyword].flatten())
        if energy_keyword:
            if energy_keyword == "energy":
                ref_energy.append(m.get_potential_energy())
            else:
                ref_energy.append(m.info[energy_keyword])
        if forces_keyword:
            if forces_keyword == "forces":
                ref_forces.extend(m.get_forces().flatten())
            else:
                ref_forces.extend(m.arrays[forces_keyword].flatten())
        if quadrupole_keyword and quadrupole_keyword in m.info:
            ref_quadrupoles.extend(m.info[quadrupole_keyword].flatten())
    result = {}
    result["energy"] = np.array(ref_energy)
    result["forces"] = np.array(ref_forces)
    if len(ref_dipoles) > 0:
        result["dipole"] = np.array(ref_dipoles)
    if len(ref_quadrupoles) > 0:
        result["quadrupole"] = np.array(ref_quadrupoles)
    return result

def plot_data(ref_data, pred_data, key, xlabel, ylabel, filename):
    """Generic function to plot reference vs predicted data."""
    plt.scatter(ref_data[key], pred_data[key], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data[key], ref_data[key], color="black", label='Identity Line')  # Identity line
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    args = parse_args()
    amp_mols = read(args.geoms, format="extxyz", index=":")
    ref_data = get_ref(amp_mols, REF_ENERGY_KEY, REF_FORCES_KEY, REF_DIPOLE_KEY, REF_QUADRUPOLE_KEY)
    AMP_data = get_ref(amp_mols, PRED_ENERGY_KEY, PRED_FORCES_KEY, PRED_DIPOLE_KEY, PRED_QUADRUPOLE_KEY)

    for name, data in zip(["Ref", "AMP"], [ref_data, AMP_data]):
        for key, value in data.items():
            print(f"{name} {key}: {value.shape} Min Max: {np.min(value): .1f} {np.max(value): .1f}")

    plot_data(ref_data, AMP_data, "energy", "Ref energy", "AMP energy", "AMPenergy.png")
    plot_data(ref_data, AMP_data, "forces", "Ref forces", "AMP forces", "AMPforces.png")
    if "dipole" in ref_data and "dipole" in AMP_data:
        plot_data(ref_data, AMP_data, "dipole", "Ref dipole", "AMP dipole", "AMPdipole.png")
    if "quadrupole" in ref_data and "quadrupole" in AMP_data:
        plot_data(ref_data, AMP_data, "quadrupole", "Ref quadrupole", "AMP quadrupole", "AMPquadrupole.png")

if __name__ == "__main__":
    main()