import torch
from torch import Tensor
import torch.nn as nn
import torchlayers as tl

import numpy as np

def A(x, k: int = -1):
    return x.unsqueeze(dim=k)


def S(x, y):
    return (x * y).sum(dim=-1, keepdim=True)

def envelope(R1):
    p = 6
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    env_val = 1.0 / R1 + a * R1 ** (p - 1) + b * R1**p + c * R1 ** (p + 1)
    return torch.where(R1 < 1, env_val, 0)


def switch(R1, r_switch: float = 4.0, r_cutoff: float = 5.0):
    X = (R1 - r_switch) / (r_cutoff - r_switch)
    X3 = torch.pow(X, 3)
    X4 = X3 * X
    X5 = X4 * X
    return torch.clip(1 - 6 * X5 + 15 * X4 - 10 * X3, 0, 1)


def build_sin_kernel(
    R1,
    cutoff: float = 5.0,
    n_kernels: int = 20,
    device: torch.device = torch.device("cuda"),
):
    d_scaled = R1 * (1.0 / cutoff)
    d_cutoff = envelope(d_scaled)
    FREQUENCIES = np.pi * torch.arange(1, n_kernels + 1).unsqueeze(0).to(device)
    return d_cutoff * torch.sin(FREQUENCIES * d_scaled), d_cutoff


def pdist(qm_coordinates: Tensor):
    indices_triu = torch.triu_indices(
        qm_coordinates.shape[1], qm_coordinates.shape[1], offset=1
    )  # .to(device=device)
    distance_matrix_norm = torch.zeros(
        (qm_coordinates.shape[0], qm_coordinates.shape[1], qm_coordinates.shape[1]),
        device=qm_coordinates.device,
    )
    norm = torch.linalg.vector_norm(
        qm_coordinates[:, indices_triu[0]] - qm_coordinates[:, indices_triu[1]], dim=-1
    )
    distance_matrix_norm[:, indices_triu[0], indices_triu[1]] = norm
    return distance_matrix_norm + distance_matrix_norm.permute([0, 2, 1])


# @torch.jit.script
# def pdist(A: Tensor):
#    A_norm = torch.square(A).sum(dim=-1, keepdim=True)
#    B_norm = A_norm.transpose(2, 1)
#    D = A_norm - 2 * torch.bmm(A, A.permute(0, 2, 1)) + B_norm
#    return torch.where(D > 0.0, torch.sqrt(D), 0.0)
# return torch.sqrt(torch.clip(D - torch.eye(A.shape[1])[None], 0.0))


def cdist(
    A: Tensor,
    B: Tensor,
):
    A_norm = torch.square(A).sum(dim=-1, keepdim=True)
    B_norm = torch.square(B).sum(dim=-1, keepdim=True).transpose(2, 1)
    D = A_norm - 2 * torch.bmm(A, B.permute(0, 2, 1)) + B_norm
    return torch.sqrt(torch.clip(D, 0.0))


def detrace(RxR):
    diagonal = torch.tile(RxR.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True), (1, 3))
    return RxR - torch.diag_embed(diagonal)


def build_Rx2(Rx1):
    return detrace(A(Rx1, -1) * A(Rx1, -2))


def prepare_features_qmmm(
    distance_matrix,
    qm_coordinates,
    mm_coordinates,
    cutoff_lr: float = 10.0,
    n_kernels: int = 20,
    device: torch.device = torch.device("cuda"),
):
    indices_qmmm = torch.where(distance_matrix < cutoff_lr)
    batch_indices, receivers_qmmm, senders_qmmm = indices_qmmm
    qm_indices = torch.stack((batch_indices, receivers_qmmm), dim=0)
    mm_indices = torch.stack((batch_indices, senders_qmmm), dim=0)
    coords_1 = qm_coordinates[batch_indices, receivers_qmmm]
    coords_2 = mm_coordinates[batch_indices, senders_qmmm]
    R1_qmmm = distance_matrix[batch_indices, receivers_qmmm, senders_qmmm].unsqueeze(-1)
    Rx1_qmmm = (coords_2 - coords_1) / R1_qmmm  # Normalized Directional Vector
    Rx2_qmmm = build_Rx2(Rx1_qmmm)  # Detrace Outer Product
    edges_qmmm, envelope = build_sin_kernel(
        R1_qmmm, cutoff_lr, n_kernels=n_kernels, device=device
    )
    receivers_qmmm = (
        indices_qmmm[1] + indices_qmmm[0] * qm_coordinates.shape[1]
    )  # Indices of atoms in the QM Zone, unidirectional interaction
    return (
        R1_qmmm,
        Rx1_qmmm,
        Rx2_qmmm,
        edges_qmmm,
        envelope,
        (qm_indices, mm_indices, receivers_qmmm),
    )


def prepare_features_esp(
    distance_matrix,
    qm_coordinates,
    mm_coordinates,
    cutoff_esp: float = 500.0,
    device: torch.device = torch.device("cuda"),
):
    indices_esp = torch.where(distance_matrix < cutoff_esp)
    batch_indices_esp, receivers_esp, senders_esp = indices_esp
    qm_indices_esp = torch.stack((batch_indices_esp, receivers_esp), dim=0)
    mm_indices_esp = torch.stack((batch_indices_esp, senders_esp), dim=0)
    coords_1 = qm_coordinates[batch_indices_esp, receivers_esp]
    coords_2 = mm_coordinates[batch_indices_esp, senders_esp]
    R1_esp = distance_matrix[batch_indices_esp, receivers_esp, senders_esp].unsqueeze(
        -1
    )
    Rx1_esp = coords_2 - coords_1
    Rx2_esp = build_Rx2(Rx1_esp)  # Detrace Outer Product
    receivers_esp = (
        indices_esp[1] + indices_esp[0] * qm_coordinates.shape[1]
    )  # Indices of atoms in the QM Zone, unidirectional interaction
    return R1_esp, Rx1_esp, Rx2_esp, (qm_indices_esp, mm_indices_esp, receivers_esp)


# Takes QM coordinates and returns information necessary for the message passing inside the QM zone.
def prepare_features_qm(
    qm_coordinates: Tensor,
    cutoff: float = 5.0,
    n_kernels: int = 20,
    device: torch.device = torch.device("cuda"),
):
    distance_matrix = pdist(qm_coordinates)
    # distance_matrix = cdist(qm_coordinates, qm_coordinates)
    # distance_matrix = cdist(qm_coordinates, qm_coordinates, device=device)
    # distance_matrix = torch.cdist(qm_coordinates, qm_coordinates)
    indices = torch.where(
        torch.logical_and(distance_matrix < cutoff, distance_matrix > 0.0)
    )
    # indices = torch.where(distance_matrix < cutoff)
    mol_id, senders, receivers = indices
    coords_1, coords_2 = (
        qm_coordinates[mol_id, senders],
        qm_coordinates[mol_id, receivers],
    )
    R1 = A(distance_matrix[mol_id, senders, receivers])
    Rx1 = (coords_2 - coords_1) / R1  # Normalized Directional Vector
    Rx2 = build_Rx2(Rx1)
    edges, envelope = build_sin_kernel(R1, cutoff, n_kernels=n_kernels, device=device)
    trius = torch.triu_indices(
        int(distance_matrix.shape[1]), int(distance_matrix.shape[1]), offset=1, device=device
    )
    R1_intra = distance_matrix[:, trius[0], trius[1]]
    return (
        R1,
        Rx1,
        Rx2,
        edges,
        envelope,
        (mol_id, senders, receivers),
        (R1_intra, trius[0], trius[1]),
    )


def build_graph(
    qm_coordinates,
    mm_coordinates,
    qm_types,
    mm_types,
    mol_charge: float,
    cutoff: float = 5.0,
    cutoff_lr: float = 10.0,
    cutoff_esp: float = 500.0,
    n_kernels: int = 20,
    n_kernels_qmmm: int = 20,
    device: torch.device = torch.device("cuda"),
):
    n_molecules, mol_size = qm_coordinates.shape[:2]
    Z = qm_types.unsqueeze(0).expand(n_molecules, -1)
    qm_types = (
        torch.nn.functional.one_hot(qm_types, num_classes=-1)
        .type(qm_coordinates.type())
        .to(device)
    )
    qm_types = qm_types.tile([n_molecules, 1, 1]).reshape((n_molecules * mol_size, -1))
    n_node = torch.full([qm_types.shape[0]], 1, dtype=torch.int64, device=device)
    distance_matrix = cdist(qm_coordinates, mm_coordinates)
    # distance_matrix = cdist(qm_coordinates, mm_coordinates, device=device)
    # distance_matrix = torch.cdist(qm_coordinates, mm_coordinates)
    (
        R1,
        Rx1,
        Rx2,
        edges,
        envelope_qm,
        (mol_id, senders, receivers),
        (R1_intra, intra_index_1, intra_index_2),
    ) = prepare_features_qm(
        qm_coordinates, cutoff=cutoff, n_kernels=n_kernels, device=device
    )
    (
        R1_qmmm,
        Rx1_qmmm,
        Rx2_qmmm,
        edges_qmmm,
        envelope_qmmm,
        (qm_indices, mm_indices, receivers_qmmm),
    ) = prepare_features_qmmm(
        distance_matrix,
        qm_coordinates,
        mm_coordinates,
        cutoff_lr=cutoff_lr,
        n_kernels=n_kernels_qmmm,
        device=device,
    )
    (
        R1_esp,
        Rx1_esp,
        Rx2_esp,
        (qm_indices_esp, mm_indices_esp, receivers_esp),
    ) = prepare_features_esp(
        distance_matrix,
        qm_coordinates,
        mm_coordinates,
        cutoff_esp=cutoff_esp,
        device=device,
    )
    torch.max(mm_indices_esp[1])
    mm_monos_esp = mm_types[mm_indices_esp[0], mm_indices_esp[1]]
    mm_monos_qmmm = mm_types[mm_indices[0], mm_indices[1]]
    if len(mm_monos_esp.shape) == 1:
        mm_monos_esp = mm_monos_esp.unsqueeze(-1)
        mm_monos_qmmm = mm_monos_qmmm.unsqueeze(-1)
    shift = mol_size * mol_id
    senders, receivers = senders + shift, receivers + shift
    graph = {}
    graph["Z"] = Z
    graph["nodes"] = qm_types
    graph["nodes0"] = qm_types
    graph["n_node"] = n_node
    graph["n_segment"] = n_node.sum()
    graph["edges"] = edges
    graph["senders"] = senders
    graph["receivers"] = receivers
    graph["edges_qmmm"] = edges_qmmm
    graph["receivers_qmmm"] = receivers_qmmm
    graph["receivers_esp"] = receivers_esp
    graph["R1"], graph["Rx1"], graph["Rx2"] = R1, A(Rx1, 1), A(Rx2, 1)
    graph["R1_qmmm"], graph["Rx1_qmmm"], graph["Rx2_qmmm"] = R1_qmmm, Rx1_qmmm, Rx2_qmmm
    graph["R1_esp"], graph["Rx1_esp"], graph["Rx2_esp"] = R1_esp, Rx1_esp, Rx2_esp
    graph["batch_size"] = torch.tensor([n_molecules], device=device)
    # graph['mol_size'] = torch.full([n_molecules, 1], mol_size, dtype=torch.float32, device=device)
    graph["mol_size"] = torch.tensor([mol_size], device=device)
    graph["mol_charge"] = torch.full([n_molecules, 1], mol_charge, dtype=torch.float32, device=device)
    graph["batch_indices"] = mol_id
    graph["qm_indices"] = qm_indices
    graph["mm_indices"] = mm_indices
    graph["qm_indices_esp"] = qm_indices_esp
    graph["mm_indices_esp"] = mm_indices_esp
    graph["mm_monos_esp"] = mm_monos_esp
    graph["mm_monos_qmmm"] = mm_monos_qmmm
    graph["monos"] = torch.zeros((int(graph["n_segment"]), 1), device=device)
    graph["dipos"] = torch.zeros((int(graph["n_segment"]), 3), device=device)
    graph["quads"] = torch.zeros((int(graph["n_segment"]), 3, 3), device=device)
    graph["qm_coordinates"] = qm_coordinates
    graph["envelope_qm"] = envelope_qm  # / cutoff
    graph["envelope_qmmm"] = envelope_qmmm  # / cutoff_lr
    graph["intra_index_1"] = intra_index_1
    graph["intra_index_2"] = intra_index_2
    graph["R1_intra"] = R1_intra
    return graph


def ff_module(
    node_size,
    num_layers,
    activation=nn.SiLU(),
    with_bias=True,
    output_size=None,
    final_activation=None,
):  # , device=torch.device('cuda')
    layers = []
    for _ in range(num_layers):
        layers.append(tl.Linear(node_size, bias=with_bias))  # .to(device)
        layers.append(activation)
    if output_size is not None:
        layers.append(tl.Linear(output_size, bias=False))  # .to(device)
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)  # .to(device)