import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.distributions import Normal
from copy import deepcopy
from evae import EGGNEncoder, AE

from data import get_graph_dataloaders, Converter
from ase import Atoms
from ase.io import write
from ase.visualize import view

a2g = Converter(radius=4.5)


def g2a(g):
    n = g.num_nodes
    return Atoms(f"C{n}", positions=g.x.detach().numpy(), cell=np.diag([4] * 3))


n_hidden = 64
encoder = EGGNEncoder(hidden_channels=n_hidden, n_layers=4)
decoder = EGGNEncoder(
    hidden_channels=n_hidden, n_layers=4, embed=False, use_edge_attr=False
)

model = AE(encoder, decoder)
# # Load model:
model.load_state_dict(torch.load("best-ae2-phi_inf-3-model.pth"))
with torch.no_grad():
    all = []
    n = 4
    h = torch.randn(n, n_hidden)
    h /= h.norm(dim=-1, keepdim=True)
    for i in range(100):
        a = Atoms(f"C{n}", positions=np.random.randn(n, 3), cell=np.diag([4] * 3), pbc=True)
        a.wrap()
    
        graph_z = a2g.convert(a.copy())
        graph_z.h = h.clone()*i
    
        graph_hat = model.decoder(graph_z.clone())
        all.append(g2a(graph_hat))

    view(all)
