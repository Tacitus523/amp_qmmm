#!/lustre/home/ka/ka_ipc/ka_he8978/miniconda3/envs/mace_env/bin/python3.12
import argparse
from ase.io import read
import os
import numpy as np
import matplotlib.pyplot as plt

# REF_GEOMS = "/lustre/work/ws/ws1/ka_he8978-dipeptide/training_data/B3LYP_aug-cc-pVTZ_vacuum/geoms.extxyz"
AMP_GEOMS = "results/amp_qmmm_geoms.extxyz" # Should already contain reference and AMP data

PLOT_ENERGY = True
PLOT_FORCES = True 
PLOT_DIPOLE = True
PLOT_QUADRUPOLE = True

# The following keywords are used to extract the data from the ASE atoms object
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
    args = parser.parse_args()
    return args

def get_ref(
        mols,
        energy_keyword=None,
        forces_keyword=None,
        dipole_keyword=None,
        quadrupole_keyword=None):
    
    ref_energy = []
    ref_forces = []
    ref_dipoles = []
    ref_quadrupoles = []
    for m in mols:
        if dipole_keyword != None:
                ref_dipoles.extend(m.info[dipole_keyword].flatten())
        if energy_keyword != None:
            if energy_keyword == "energy":
                ref_energy.append(m.get_potential_energy())
            else:
                ref_energy.append(m.info[energy_keyword])
        if forces_keyword != None:
            if forces_keyword == "forces":
                ref_forces.extend(m.get_forces().flatten())
            else:
                ref_forces.extend(m.arrays[forces_keyword].flatten())
        if quadrupole_keyword != None:
            ref_quadrupoles.extend(m.info[quadrupole_keyword].flatten())
    ref_energy = np.array(ref_energy)
    ref_forces = np.array(ref_forces)
    ref_dipoles = np.array(ref_dipoles)
    ref_quadrupoles = np.array(ref_quadrupoles)
    return {
        "energy": ref_energy, 
        "forces": ref_forces,
        "dipole": ref_dipoles,
        "quadrupole": ref_quadrupoles,
        }
args = parse_args()
amp_mols = read(args.geoms, format="extxyz", index=":")
ref_data = get_ref(amp_mols, REF_ENERGY_KEY, REF_FORCES_KEY, REF_DIPOLE_KEY, REF_QUADRUPOLE_KEY)
AMP_data = get_ref(amp_mols, PRED_ENERGY_KEY, PRED_FORCES_KEY, PRED_DIPOLE_KEY, PRED_QUADRUPOLE_KEY)

if PLOT_ENERGY:
    plt.scatter(ref_data["energy"], AMP_data["energy"], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data["energy"], ref_data["energy"], color="black", label='Identity Line')  # Identity line
    plt.xlabel('Ref energy')  # X-axis Label
    plt.ylabel('AMP energy')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("AMPenergy.png", dpi=300)
    # plt.show()
    plt.close()

if PLOT_FORCES:
    plt.scatter(ref_data["forces"], AMP_data["forces"], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data["forces"], ref_data["forces"], color="black", label='Identity Line')  # Identity line
    plt.xlabel('Ref forces')  # X-axis Label
    plt.ylabel('AMP forces')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("AMPforces.png", dpi=300)
    #plt.show()
    plt.close()

if PLOT_DIPOLE:
    plt.scatter(ref_data["dipole"], AMP_data["dipole"], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data["dipole"], ref_data["dipole"], color="black", label='Identity Line')  # Identity line
    plt.xlabel('Ref dipole')  # X-axis Label
    plt.ylabel('AMP dipole')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("AMPdipole.png",dpi=300)
    #plt.show()
    plt.close()

    
if PLOT_QUADRUPOLE:
    plt.scatter(ref_data["quadrupole"], AMP_data["quadrupole"], c='blue', alpha=0.5, label='Data Points')  # Scatter plot
    plt.plot(ref_data["quadrupole"], ref_data["quadrupole"], color="black", label='Identity Line')  # Identity line
    plt.xlabel('Ref quadrupole')  # X-axis Label
    plt.ylabel('AMP quadrupole')  # Y-axis Label
    plt.tight_layout()  # Tight layout for nicer appearance
    plt.savefig("AMPquadrupole.png", dpi=300)
    #plt.show()
    plt.close()



