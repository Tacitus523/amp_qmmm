#!/lustre/home/ka/ka_ipc/ka_he8978/miniconda3/envs/mace_env/bin/python3.12
import argparse
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt

# Default geometry file
AMP_GEOMS = "results/amp_qmmm_geoms.extxyz"

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
    parser.add_argument("-g", "--geoms", type=str, default=AMP_GEOMS, help="Path to the geometry file")
    return parser.parse_args()

def get_ref(mols, energy_keyword=None, forces_keyword=None, dipole_keyword=None, quadrupole_keyword=None):
    ref_energy = []
    ref_forces = []
    ref_dipoles = []
    ref_quadrupoles = []
    for m in mols:
        if dipole_keyword:
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
        if quadrupole_keyword:
            ref_quadrupoles.extend(m.info[quadrupole_keyword].flatten())
    return {
        "energy": np.array(ref_energy),
        "forces": np.array(ref_forces),
        "dipole": np.array(ref_dipoles),
        "quadrupole": np.array(ref_quadrupoles),
    }

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

    plot_data(ref_data, AMP_data, "energy", "Ref energy", "AMP energy", "AMPenergy.png")
    plot_data(ref_data, AMP_data, "forces", "Ref forces", "AMP forces", "AMPforces.png")
    plot_data(ref_data, AMP_data, "dipole", "Ref dipole", "AMP dipole", "AMPdipole.png")
    plot_data(ref_data, AMP_data, "quadrupole", "Ref quadrupole", "AMP quadrupole", "AMPquadrupole.png")

if __name__ == "__main__":
    main()