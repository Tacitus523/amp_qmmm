#!/usr/bin/env python
import argparse
import os

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from AMPQMMM import AMPQMMM as AMPQMMM_precision
from AMPQMMMmin import AMPQMMM as AMPQMMM_min

DEFAULT_OUTPUT_MODEL_NAME: str = "wrapped_{input_model}.pt"
DEFAULT_OUTPUT_STATEDICT_NAME: str = "wrapped_{input_model}_state_dict.pth"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrap an AMP model for inference. Input and output dictionary instead of list")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the AMP model file.")
    #parser.add_argument("-o", "--output", type=str, required=False, default=DEFAULT_OUTPUT, help="Path to the output file.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file {args.model} does not exist.")

    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    return args

def wrap_model(model_path: str) -> None:

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

def main():
    args = parse_args()
    wrap_model(args.model)


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
                    - qm_gradients: Tensor of shape (N, 3) containing the QM gradients.
                    - forces: Tensor of shape (N, 3) containing the forces.
                    - mm_gradients: Tensor of shape (N, 3) containing the MM gradients.
                    - mm_forces: Tensor of shape (N, 3) containing the MM forces.
                    - dipoles: Tensor of shape (N, 3) containing the dipoles.
                    - quadrupoles: Tensor of shape (N, 3) containing the quadrupoles.
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
            create_graph=True,
            retain_graph=True
        )[0] # shape (batch, N_QM, 3)

        mm_grad_output: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
        mm_gradients = torch.autograd.grad(
            outputs=[energy],
            inputs=[batch_inputs["mm_coordinates"]],
            grad_outputs=mm_grad_output,
            create_graph=True,
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
        return output

    def batch_to_input(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert the input dictionary to a tuple of tensors"""

        # check if all atoms are identical (only one molecule)
        #assert torch.all(batch["qm_charges"] == batch["qm_charges"][0]), "QM charges are not identical"
        qm_charges = batch["qm_charges"]
        qm_coordinates = batch["qm_coordinates"]
        mm_charges = batch["mm_charges"]
        mm_coordinates = batch["mm_coordinates"]
        dummy = torch.zeros_like(qm_charges)
        return (qm_charges, qm_coordinates, dummy, mm_charges, mm_coordinates)
    


if __name__ == "__main__":
    main()