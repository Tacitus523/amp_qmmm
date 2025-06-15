#!/usr/bin/env python
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

from ase.io import read
from ase import Atoms
import numpy as np
import torch

from AMPQMMM import AMPQMMM as AMPQMMM_precision
from AMPQMMMmin import AMPQMMM as AMPQMMM_min

sys.path.append("/home/ka/ka_ipc/ka_he8978/bin/amp_qmmm_scripts")
from utils_hdf5 import extract_mm_data # type: ignore

DEFAULT_OUTPUT_MODEL_NAME: str = "wrapped_{input_model}.pt"
DEFAULT_OUTPUT_STATEDICT_NAME: str = "wrapped_{input_model}_state_dict.pth"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrap an AMP model for inference. Input and output dictionary instead of list")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the AMP model file.")
    parser.add_argument("-g", "--geoms", type=str, required=False, default=None, help="Path to the test data. Optional. Can be used to test the model equivalency after wrapping. Format: .extxyz")
    parser.add_argument("--pc", type=str, default=None, help="Path to the concatenated pointcharges files. Optional")
    #parser.add_argument("-o", "--output", type=str, required=False, default=DEFAULT_OUTPUT, help="Path to the output file.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file {args.model} does not exist.")

    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    return args

class WrappedAMPModel(torch.nn.Module):
    def __init__(self, model: AMPQMMM_min|AMPQMMM_precision) -> None:
        super(WrappedAMPModel, self).__init__()
        self.model = model

    def forward(self, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Transforms the input dictionary to a tuple of tensors and calls the original model's forward method.

        Args:
            batch_inputs (Dict[str, torch.Tensor]): Input dictionary containing the model inputs.
                Keys:
                    - qm_charges: Tensor of shape (N,) containing the QM charges.
                    - qm_coordinates: Tensor of shape (N, 3) containing the QM coordinates. Requires gradients.
                    - mm_charges: Tensor of shape (N,) containing the MM charges.
                    - mm_coordinates: Tensor of shape (N, 3) containing the MM coordinates. Requires gradients.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary containing the model outputs.
                Keys:
                    - gradients: Tensor of shape (N, 3) containing the QM gradients.
                    - forces: Tensor of shape (N, 3) containing the forces.
                    - mm_gradients: Tensor of shape (N, 3) containing the MM gradients.
                    - mm_forces: Tensor of shape (N, 3) containing the MM forces.
                    - dipoles: Tensor of shape (N, 3) containing the dipoles.
                    - quadrupoles: Tensor of shape (N, 6) containing the quadrupoles.
        """
        # Mark the input tensors as requiring gradients
        batch_inputs["qm_coordinates"].requires_grad_(True)
        batch_inputs["mm_coordinates"].requires_grad_(True)

        # Convert the input dictionary to a tuple of tensors
        inputs = self.batch_to_input(batch_inputs)

        energy, graph = self.model.forward_with_graph(inputs)
        dipole = self.model._molecular_dipole(graph)
        quadrupole = self.model._molecular_quadrupole(graph)

        qm_grad_output: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
        qm_gradients = torch.autograd.grad(
            outputs=[energy],
            inputs=[batch_inputs["qm_coordinates"]],
            grad_outputs=qm_grad_output,
            create_graph=False,
            retain_graph=True
        )[0] # shape (batch, N_QM, 3)

        mm_grad_output: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
        mm_gradients = torch.autograd.grad(
            outputs=[energy],
            inputs=[batch_inputs["mm_coordinates"]],
            grad_outputs=mm_grad_output,
            create_graph=False,
            retain_graph=True
        )[0] # shape (batch, N_MM, 3)

        if qm_gradients is None:
            raise ValueError("QM gradients are None. Check the model and input data.")

        if mm_gradients is None:
            raise ValueError("MM gradients are None. Check the model and input data.")

        # Create the output dictionary
        output = {
            "energy": energy,
            "gradients": qm_gradients,
            "forces": -1*qm_gradients,
            "mm_gradients": mm_gradients,
            "mm_forces": -1*mm_gradients,
            "dipoles": dipole,
            "quadrupoles": quadrupole
        }
        # for key, value in batch_inputs.items():
        #     print(f"{key}: {value}")
        # for key, value in output.items():
        #     print(f"{key}: {value}")
        return output

    def batch_to_input(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, torch.Tensor]:
        """Convert the input dictionary to a tuple of tensors"""

        # check if all atoms are identical (only one molecule)
        #assert torch.all(batch["qm_charges"] == batch["qm_charges"][0]), "QM charges are not identical"
        qm_charges = batch["qm_charges"][0]
        qm_coordinates = batch["qm_coordinates"]
        mm_charges = batch["mm_charges"]
        mm_coordinates = batch["mm_coordinates"]

        return (qm_charges, qm_coordinates, None, mm_charges, mm_coordinates)
 
def wrap_model(model_path: str) -> WrappedAMPModel:
    # Load the model
    model = torch.load(model_path, map_location="cpu")
    wrapped_model = WrappedAMPModel(model)


    # Save the wrapped model
    model_name = os.path.basename(model_path)
    output_dir = os.path.abspath(os.path.dirname(model_path))
    model_name = model_name.split(".")[0]
    output_model_name = DEFAULT_OUTPUT_MODEL_NAME.format(input_model=model_name)
    output_state_dict_name = DEFAULT_OUTPUT_STATEDICT_NAME.format(input_model=model_name)
    output_model_path = os.path.join(output_dir, output_model_name)
    output_state_dict_path = os.path.join(output_dir, output_state_dict_name)

    wrapped_state_dict = wrapped_model.state_dict()
    unwrapped_state_dict = {k.replace("model.", ""): v for k, v in wrapped_state_dict.items()}

    torch.save(wrapped_model, output_model_path)
    print(f"Wrapped model saved to {output_model_path}")
    torch.save(unwrapped_state_dict, output_state_dict_path)
    print(f"Wrapped model state dict saved to {output_state_dict_path}")
    return wrapped_model

def test_wrapped_model(wrapped_model: WrappedAMPModel, args: argparse.Namespace) -> None:
    """Test the wrapped model with provided geometries and point charges.
    Kinda pointless, because only function from the wrapped model are used.
    Compare to results from a test_amp.py script run to get reliable results.
    """

    geom_path: Optional[str] = args.geoms
    pc_path: Optional[str] = args.pc

    if geom_path is None and pc_path is None:
        print("No geometry or point charge files provided. Skipping test.")
        return
    if geom_path is None or pc_path is None:
        raise ValueError("Geometry and point charge files must be provided for testing.")
    
    if not os.path.exists(geom_path):
        raise FileNotFoundError(f"Geometry file {geom_path} does not exist.")
    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"Point charge file {pc_path} does not exist.")
    
    geoms: List[Atoms] = read(geom_path, ":")
    qm_coordinates: List[np.ndarray] = [molecule.get_positions() for molecule in geoms]
    qm_charges: List[np.ndarray] = [molecule.get_atomic_numbers() for molecule in geoms]

    mm_charges, mm_coordinates, _ = extract_mm_data(pc_path)
    if len(qm_coordinates) != len(mm_charges) or len(qm_coordinates) != len(mm_coordinates):
        raise ValueError("Mismatch in number of geometries and point charges/coordinates.")
    
    # Batch the inputs
    batch_inputs: Dict[str, torch.Tensor] = {
        "qm_charges": torch.tensor(qm_charges),
        "qm_coordinates": torch.tensor(qm_coordinates),
        "mm_charges": torch.tensor(mm_charges),
        "mm_coordinates": torch.tensor(mm_coordinates)
    }

    # Run the wrapped model in batches
    batch_size = 10
    num_batches = len(qm_coordinates) // batch_size + (1 if len(qm_coordinates) % batch_size > 0 else 0)
    wrapped_output_dicts: List[Dict[str, torch.Tensor]] = []
    unwrapped_output_dicts: List[Dict[str, torch.Tensor]] = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(qm_coordinates))
        batch = {key: value[start_idx:end_idx] for key, value in batch_inputs.items()}
        wrapped_output_dict: Dict[str, torch.Tensor] = wrapped_model(batch)
        wrapped_output_dicts.append(wrapped_output_dict)

        # Redo for unwrapped model
        inputs = wrapped_model.batch_to_input(batch)

        energy, graph = wrapped_model.model.forward_with_graph(inputs)
        qm_gradients = torch.autograd.grad(
            outputs=[energy],
            inputs=[inputs[1]],  # qm_coordinates
            grad_outputs=[torch.ones_like(energy)],
            create_graph=False,
            retain_graph=True
        )[0]
        mm_gradients = torch.autograd.grad(
            outputs=[energy],
            inputs=[inputs[4]],  # mm_coordinates
            grad_outputs=[torch.ones_like(energy)],
            create_graph=False,
            retain_graph=True
        )[0]
        dipole = wrapped_model.model._molecular_dipole(graph)
        quadrupole = wrapped_model.model._molecular_quadrupole(graph)
        

        unwrapped_output_dict: Dict[str, torch.Tensor] = {
            "energy": energy,
            "gradients": qm_gradients,
            "forces": -1 * qm_gradients,
            "mm_gradients": mm_gradients,
            "mm_forces": -1 * mm_gradients,
            "dipoles": dipole,
            "quadrupoles": quadrupole
        }
        unwrapped_output_dicts.append(unwrapped_output_dict)

    # Concatenate the outputs
    wrapped_concatenated_output: Dict[str, torch.Tensor] = {}
    unwrapped_concatenated_output: Dict[str, torch.Tensor] = {}
    for key in wrapped_output_dicts[0].keys():
        wrapped_concatenated_output[key] = torch.cat([output_dict[key] for output_dict in wrapped_output_dicts], dim=0)
        unwrapped_concatenated_output[key] = torch.cat([output_dict[key] for output_dict in unwrapped_output_dicts], dim=0)

    # # Print the outputs
    # print("Test results:")
    # for key, value in wrapped_concatenated_output.items():
    #     if "mm_" in key:
    #         continue
    #     print(f"{key}: {value.shape}")
    #     print(value)

    # Check if the outputs are equivalent
    for key in wrapped_concatenated_output.keys():
        assert torch.allclose(
            wrapped_concatenated_output[key],
            unwrapped_concatenated_output[key],
            rtol=1e-5,
            atol=1e-8
        ), f"Outputs for {key} are not equivalent after wrapping."
    print("All outputs are equivalent after wrapping.")

def main():
    args: argparse.Namespace = parse_args()
    wrapped_model: WrappedAMPModel = wrap_model(args.model)
    test_wrapped_model(wrapped_model, args)

if __name__ == "__main__":
    main()