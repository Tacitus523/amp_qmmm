from AMPHelpers import S, A, build_graph, build_Rx2, ff_module

import torch
import torch.nn as nn
import torchlayers as tl
import os

from typing import Dict, Tuple
from torch import Tensor
from torch_scatter import scatter

class AMPQMMM(nn.Module):
    def __init__(self, activation=nn.SiLU(), **kwargs):
        super(AMPQMMM, self).__init__()
        self.activation = activation
        if not "mol_charge" in kwargs:
            self.mol_charge = 0.0
        self.__dict__.update(kwargs)
        self.a, self.b, self.c = 6.0, 15.0, 10.0
        self.register_buffer("element_masses", torch.load(os.path.join(os.path.dirname(__file__), "constants", "element_masses.pt")))
        
        self.embedding_nodes = tl.Linear(self.node_size, bias=False)
        self.embedding_edges = tl.Linear(self.node_size // 4, bias=False)    
        
        self.in_update_layers = nn.ModuleList([ff_module(self.node_size, 2, activation=self.activation) for _ in range(self.n_steps)])
        self.in_message_layers = nn.ModuleList([ff_module(self.node_size, 2, activation=self.activation) for _ in range(self.n_steps)])
        self.eq_message_layers = nn.ModuleList([ff_module(self.node_size, 1, output_size=(self.order + 1) * self.n_channels, activation=self.activation) for _ in range(self.n_steps)])
        
        self.QMMM_potential = ff_module(self.node_size, 2, output_size=1, activation=self.activation) 
        self.QMMM_density = ff_module(self.node_size, 2, output_size=1, activation=self.activation) 
        self.QM_alpha = ff_module(self.node_size // 2, 1, output_size=self.order * self.n_channels, activation=self.activation, final_activation=nn.Softplus()) 
        self.B_coefficients = ff_module(8, 1, output_size=self.order, activation=self.activation)
        if self.aniso_esp:
            self.QM_coefficients = ff_module(self.node_size, 1, output_size=self.order + 1, activation=self.activation)

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        return self._run(inputs)[0]
    
    def forward_with_graph(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        return self._run(inputs)
    
    def forward_with_molecular_dipole(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        potential_energy, graph = self._run(inputs)
        return potential_energy, self._molecular_dipole(graph)
    
    def _run(self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        qm_types, qm_coordinates, _, mm_charges, mm_coordinates = inputs
        graph = build_graph(qm_coordinates, mm_coordinates, qm_types, mm_charges, mol_charge=self.mol_charge,
                            cutoff=self.cutoff, cutoff_lr=self.cutoff_lr, 
                            n_kernels=self.n_kernels, n_kernels_qmmm=self.n_kernels_qmmm,
                            device=self.device)
        return self._process_graph(graph)
        
    def _process_graph(self, graph: Dict[str, torch.Tensor]):
        graph = self._embed(graph)
        graph = self._pass_messages(graph)
        return self._calculate_energy_terms(graph)
    
    def _calculate_energy_terms(self, graph: Dict[str, torch.Tensor]):
        qm_term = self.QMMM_potential(graph['nodes'])        
        qm_term = torch.reshape(qm_term, [int(graph['batch_size']), -1]).sum(-1, keepdim=True)
        coulomb_term, graph = self._calculate_coulomb_term(graph) 
        if self.delta_qm or self.delta_qmmm:
            return qm_term + coulomb_term, graph
        else:
            return qm_term + coulomb_term, graph
        
    def _calculate_coulomb_term(self, graph: Dict[str, torch.Tensor]):
        QM_charges = self.QMMM_density(graph['nodes']) * 1e-2
        QM_charges = neutralize_charges(QM_charges, graph) 
        graph = self._build_multipoles_esp(graph)     
        graph['monos'] = QM_charges # We want to use the previously predicted (charge conserving) monopoles.  
        coulomb_term = ESP_multipoles(graph)
        coulomb_term = scatter(coulomb_term, graph['qm_indices_esp'][0], dim=0)
        coulomb_terms_qm = self._coulomb_qm(graph)
        coulomb_term = (coulomb_term + coulomb_terms_qm) * 1389.35457644382 
        graph['coulomb_term'] = coulomb_term
        return coulomb_term, graph
    
    def _embed(self, graph: Dict[str, torch.Tensor]):
        features_edge = torch.cat((graph['edges'], graph['nodes'][graph['receivers']], 
                                   graph['nodes'][graph['senders']]), dim=-1)
        graph['edges'] = self.embedding_edges(features_edge)
        graph['nodes'] = self.embedding_nodes(graph['nodes'])
        graph['edge_features'] = graph['edges'].clone().detach()
        #graph['edge_features_qmmm'] = torch.cat((graph['nodes'][graph['receivers_qmmm']], graph['edges_qmmm'])
        return graph
    
    def _pass_messages(self, graph: Dict[str, torch.Tensor]):
        for step, (eq_message_layer, in_message_layer, in_update_layer) in \
            enumerate(zip(self.eq_message_layers, self.in_message_layers, self.in_update_layers)): 
            features_ij = torch.cat((graph['nodes'][graph['receivers']], graph['nodes'][graph['senders']]), dim=-1)
            edge_features = torch.cat((features_ij, graph['edge_features']), dim=-1) 
            graph = build_poles(graph, eq_message_layer(edge_features))  
            if step == (self.n_steps - 1):
                graph = self._include_mm_polarization(graph)
            aniso_feature = build_aniso_feature(graph)
            graph['edge_features'] = torch.cat((aniso_feature, graph['edges']), dim=-1)
            message_features = torch.cat((features_ij, graph['edge_features']), dim=-1)
            messages = scatter(in_message_layer(message_features) * graph['envelope_qm'], graph['receivers'], dim=0)
            graph['nodes'] = graph['nodes'] + in_update_layer(torch.cat((graph['nodes'], messages), dim=-1)) 
        return graph
        
    def _build_multipoles_esp(self, graph: Dict[str, torch.Tensor]):
        features_ij = torch.cat((graph['nodes'][graph['receivers']], graph['nodes'][graph['senders']]), dim=-1)
        QM_coefficients = self.QM_coefficients(torch.cat((features_ij, graph['edge_features']), dim=-1))
        graph = build_poles(graph, QM_coefficients)  
        graph['monos'] = graph['monos'][:, 0] 
        graph['dipos'] = (graph['dipos'][:, 0] + graph['dipos_qmmm']) * 1e-2
        graph['quads'] = (graph['quads'][:, 0] + graph['quads_qmmm']) * 1e-2
        return graph
    
    def _coulomb_qm(self, graph: Dict[str, Tensor]):
        switching_esp = self._switching_fn0(graph["R1_intra"])
        coulomb_weights = switching_esp * torch.reciprocal(graph["R1_intra"])
        monos = graph["monos"].reshape(graph["batch_size"].item(), graph["mol_size"].item()) 
        monos_1 = torch.index_select(monos, dim=1, index=graph["intra_index_1"])
        monos_2 = torch.index_select(monos, dim=1, index=graph["intra_index_2"])
        coulomb_term = coulomb_weights * monos_1 * monos_2
        return coulomb_term.sum(dim=-1, keepdim=True)
    
    def _switching_fn0(self, R1):
        X = R1 / self.cutoff
        X3 = torch.pow(X, 3)
        X4 = X3 * X
        return torch.clip(self.a * X4 * X - self.b * X4 + self.c * X3, 0.0, 1.0)
        
    def _include_mm_polarization(self, graph: Dict[str, torch.Tensor]):
        QMMM_edge_features = torch.cat((QMMM_G_matrices(graph), graph['edges_qmmm']), dim=-1)       
        graph['alphas'] = self.QM_alpha(graph['nodes'])
        graph['b_coefficients'] = self.B_coefficients(QMMM_edge_features) * graph['envelope_qmmm']       
        graph['field'] = graph['mm_monos_qmmm'] / torch.square(graph['R1_qmmm'])
        coefficients = graph['b_coefficients'] * graph['field']
        graph['dipos_qmmm'] = graph['alphas'][..., 0:1] * scatter(coefficients[..., 0:1] * graph['Rx1_qmmm'], graph['receivers_qmmm'], dim=0)    
        graph['quads_qmmm'] = A(graph['alphas'][..., 1:2]) * scatter(A(coefficients[..., 1:2]) * graph['Rx2_qmmm'], graph['receivers_qmmm'], dim=0)
        graph['dipos'] = graph['dipos'] + A(graph['dipos_qmmm'], 1)
        graph['quads'] = graph['quads'] + A(graph['quads_qmmm'], 1)
        return graph        
    
    def _molecular_dipole(self, graph: Dict[str, torch.Tensor]):   
        qm_coords = graph["qm_coordinates"] - compute_com(graph["qm_coordinates"], self.element_masses[graph["Z"]])
        contribution_dipoles = graph['dipos'].reshape(qm_coords.shape)
        contribution_monopoles = graph['monos'].reshape(qm_coords.shape[:2]).unsqueeze(-1) * qm_coords
        return (contribution_dipoles + contribution_monopoles).sum(-2)
    
    def _molecular_quadrupole(self, graph: Dict[str, torch.Tensor]):
        qm_coords = graph["qm_coordinates"] - compute_com(graph["qm_coordinates"], self.element_masses[graph["Z"]])
        contribution_quadrupoles = graph['quads'].reshape((*qm_coords.shape, 3))
        monos = A(A(graph['monos'].reshape(qm_coords.shape[:2])))
        Rx2 = build_Rx2(qm_coords - qm_coords.mean(dim=1, keepdims=True))
        contribution_monopoles = (monos * Rx2)
        return (contribution_quadrupoles + contribution_monopoles).sum(dim=1)
    
def compute_com(coords, masses):
    masses = masses.reshape((coords.shape[0], coords.shape[1], 1))
    masses = masses / masses.sum(-2, keepdim=True)
    return (masses * coords).sum(-2, keepdim=True)

def build_poles(graph: Dict[str, torch.Tensor], coefficients):
    coefficients = (coefficients * graph['envelope_qm']).tensor_split(3, dim=-1)
    graph['monos'] = scatter(A(coefficients[0]), graph['receivers'], dim=0)
    graph['dipos'] = scatter(A(coefficients[1]) * graph['Rx1'], graph['receivers'], dim=0)
    graph['quads'] = scatter(A(A(coefficients[2])) * graph['Rx2'], graph['receivers'], dim=0)
    return graph

def add_QMMM_polarization(graph: Dict[str, torch.Tensor], QMMM_coefficients):
    coefficients = (QMMM_coefficients * graph['envelope_qmmm']).tensor_split(2, dim=-1)
    dipos_qmmm = scatter(A(coefficients[0]) * graph['Rx1_qmmm'], graph['receivers_qmmm'], dim=0)    
    quads_qmmm = scatter(A(A(coefficients[1])) * graph['Rx2_qmmm'], graph['receivers_qmmm'], dim=0)
    graph['dipos'] = graph['dipos'] + dipos_qmmm
    graph['quads'] = graph['quads'] + quads_qmmm        
    return graph

def build_multi_feature(graph: Dict[str, torch.Tensor]):  
    d_norm = torch.norm(graph['dipos'], dim=-1, keepdim=True)
    q_norm = A(torch.norm(graph['quads'], dim=[-1, -2]))
    return torch.cat((graph['monos'], d_norm, q_norm), dim=-1).reshape([d_norm.shape[0], -1])   

def neutralize_charges(atomic_charges, graph: Dict[str, torch.Tensor]):
    charge_residuals = atomic_charges.reshape([int(graph['batch_size']), -1]).mean(-1, keepdim=True)
    charge_residuals = charge_residuals - graph['mol_charge'] / graph['mol_size']
    charge_residuals = charge_residuals.tile((1, int(graph['mol_size']))).reshape([-1, 1])
    return atomic_charges - charge_residuals

def G_matrices_2(graph: Dict[str, torch.Tensor]):
    monos_1, monos_2 = graph['monos'][graph['senders']], graph['monos'][graph['receivers']]
    dipos_1, dipos_2 = graph['dipos'][graph['senders']], graph['dipos'][graph['receivers']]
    quads_1, quads_2 = graph['quads'][graph['senders']], graph['quads'][graph['receivers']]
    D1_Rx1, D2_Rx1 = S(dipos_1, graph['Rx1']), S(dipos_2, graph['Rx1'])
    dipo_dipo = S(dipos_1, dipos_2) 
    Q1_Rx1 = torch.einsum('bmjk, bmk -> bmj', quads_1, graph['Rx1']) 
    Q2_Rx1 = torch.einsum('bmjk, bmk -> bmj', quads_2, graph['Rx1'])
    Q1_Rx2 = A(torch.einsum('bmjk, bmjk -> bm', quads_1, graph['Rx2'])) 
    Q2_Rx2 = A(torch.einsum('bmjk, bmjk -> bm', quads_2, graph['Rx2']))
    quad_dipo = S(Q1_Rx1, dipos_2)
    dipo_quad = S(Q2_Rx1, dipos_1)
    quad_quad = A(torch.einsum('bmjk, bmjk -> bm', quads_1, quads_2)) 
    quad_R = S(Q1_Rx1, Q2_Rx1)
    return torch.cat((monos_1, monos_2, 
                      D1_Rx1, D2_Rx1, dipo_dipo,
                      Q1_Rx2, Q2_Rx2, quad_dipo, dipo_quad, quad_quad, quad_R), dim=-1).reshape([monos_1.shape[0], -1])

def G_matrices_ESP(graph: Dict[str, torch.Tensor]):
    Rx1 = graph['Rx1_esp']
    Rx2 = graph['Rx2_esp']
    qm_monos = graph['monos'][graph['receivers_esp']]
    qm_dipos = graph['dipos'][graph['receivers_esp']]
    qm_quads = graph['quads'][graph['receivers_esp']]
    mm_monos = graph['mm_monos_esp']
    D1_Rx1 = S(qm_dipos, Rx1)
    Q1_Rx1 = torch.einsum('ijk, ik -> ij', qm_quads, Rx1)
    Q1_Rx2 = A(torch.einsum('ijk, ijk -> i', qm_quads, Rx2))
    G0 = qm_monos * mm_monos
    G1 = D1_Rx1 * mm_monos 
    G2 = Q1_Rx2 * mm_monos   
    return G0, G1, G2
    
def B_matrices_ESP(graph: Dict[str, torch.Tensor]):
    R1 = graph['R1_esp']
    R2 = torch.square(R1)
    B0 = 1 / R1
    B1 = B0 / R2
    B2 = 3 * B1 / R2
    return B0, B1, B2

def ESP_multipoles(graph: Dict[str, torch.Tensor]):
    B0, B1, B2 = B_matrices_ESP(graph)
    G0, G1, G2 = G_matrices_ESP(graph)
    return (G0 * B0 + G1 * B1 + G2 * B2) 
    
def build_aniso_feature(graph: Dict[str, torch.Tensor]):
    return G_matrices_2(graph)

def G_matrices_2_QMMM_CHARGE(graph: Dict[str, torch.Tensor]):
    qm_dipos = graph['dipos'][graph['receivers_qmmm'], 0:1]
    qm_quads = graph['quads'][graph['receivers_qmmm'], 0:1]
    D1_Rx1 = S(qm_dipos, A(graph['Rx1_qmmm'], 1))
    Q1_Rx2 = A(torch.einsum('bmjk, bmjk -> bm', qm_quads, A(graph['Rx2_qmmm'], 1)))
    return torch.cat((D1_Rx1, Q1_Rx2), dim=-1) 

def QMMM_G_matrices(graph: Dict[str, torch.Tensor]):
    QMMM_G_features = G_matrices_2_QMMM_CHARGE(graph)
    return QMMM_G_features.reshape([QMMM_G_features.shape[0], -1])

