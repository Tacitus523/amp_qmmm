import torch
from typing import Dict, Tuple

class AtomicEnergiesModule(torch.nn.Module):
    atomic_energies: torch.Tensor
    specified_mask: torch.Tensor

    def __init__(self, atomic_energies_dict: Dict[int, float]) -> None:
        
        super().__init__()
        
        # Initialize atomic energies tensor with max size 118 (number of elements in periodic table)
        max_elements = 118
        atomic_energies = torch.zeros(max_elements + 1, dtype=torch.get_default_dtype())  # +1 for dummy at position 0
        specified_mask = torch.zeros(max_elements + 1, dtype=torch.bool)
        
        # Fill in specified energies
        for atomic_number, energy in atomic_energies_dict.items():
            if 1 <= atomic_number <= max_elements:
                atomic_energies[atomic_number] = energy
                specified_mask[atomic_number] = True
        
        self.register_buffer("atomic_energies", atomic_energies)
        self.register_buffer("specified_mask", specified_mask)
    
    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        - atomic_numbers: tensor of atomic numbers [...]
        
        Returns:
        - total_atomic_energy: sum of atomic energies for the given atomic numbers
        """
        # Check if all requested atomic numbers have a specified energy
        unique_atomic_numbers = torch.unique(atomic_numbers)
        assert self.specified_mask[unique_atomic_numbers].all(), (
            f"No energy specified for element(s) with atomic number(s): "
            f"{unique_atomic_numbers[~self.specified_mask[unique_atomic_numbers]].tolist()}"
        )
        
        # Retrieve and sum the energies
        return torch.sum(self.atomic_energies[atomic_numbers], dim=-1)

    def __repr__(self):
        formatted_energies = ", ".join([f"{i}: {e:.4f}" for i, e in enumerate(self.atomic_energies) if self.specified_mask[i]])
        return f"{self.__class__.__name__}(energies={{{formatted_energies}}})"


class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "shift", torch.tensor(shift, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )

