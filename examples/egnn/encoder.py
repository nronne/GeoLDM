import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

"""
Graph Data:
    - edge_attr: (num_edges, 1) ; edge distances
    - edge_index: (2, num_edges) ; edge indices (flattened adjancency matrix)
    - h: (num_nodes, num_scalar_featuers) ; scalar node features (updated in-place)
    - x: (num_nodes, 3) ; tensor node features (updated in-place)
    Atoms-enherited:
    - pos: (num_nodes, 3)
    - cell: (3, 3)
    - cell_offset: (3)
    - y: (1)
    - force: (num_nodes, 3)
    Awailable through mappings:
    - x_i: (num_edges, num_scalar_features)
    - x_j: (num_edges, num_scalar_features)
    - h_i: (num_edges, num_vector_features, 3)
    - h_j: (num_edges, num_vector_features, 3)
    

"""




"""
Standard GNN Convolution and Encoder

paper: https://arxiv.org/abs/1609.02907

"""

class ConvGNN(MessagePassing):
    def __init__(
        self, in_channels, out_channels, aggr="add", flow="target_to_source", **kwargs
    ):
        super(ConvGNN, self).__init__(aggr=aggr, flow=flow, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.phi_e = nn.Sequential(
            nn.Linear(in_channels * 2 + 1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.phi_h = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.lin = nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(
        self, x, edge_attr, edge_index
    ):  # x: [N, in_channels], edge_attr: [E, 1], edge_index: [2, E]
        # print("x:", x.shape)
        x = self.lin(x)
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        return x

    def message(self, x_j, x_i, edge_attr):
        combined = torch.cat([x_j, x_i, edge_attr], dim=-1)
        m_ij = self.phi_e(combined)
        return m_ij

    def update(self, aggr_out, x):
        return self.phi_h(torch.cat([aggr_out, x], dim=-1))


class GNNEncoder(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=5):
        super(GNNEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(100, hidden_dim, padding_idx=0)

        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(ConvGNN(hidden_dim, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, batch):
        # assert batch.num_graphs == 1, "only batch size 1 is supported for now"
        batch.x = self.embedding(batch.atomic_numbers.int())  # embed atomic numbers
        batch.h = torch.zeros(batch.num_nodes, self.hidden_dim, 3)
        node_attr = batch.x

        edge_attr = batch.edge_attr
        edge_index = batch.edge_index
        # print(node_attr.shape)
        # print(edge_attr.shape)
        # print(edge_index.shape)

        for i in range(self.n_layers):
            node_attr = self.convs[i](node_attr, edge_attr, edge_index)
        batch.x = node_attr

        return batch


"""
SchNet Convolution and Encoder

Paper: https://arxiv.org/abs/1706.08566

"""


class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_gaussians = num_gaussians
        self.num_filters = num_filters
        self.cutoff = cutoff

        self.conv = ConvCF(
            hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff
        )
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr, gauss_exp):
        x = self.conv(x, edge_index, edge_attr, gauss_exp)
        x = self.act(x)
        x = self.lin(x)
        return x


class ConvCF(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_filters,
        num_gaussians,
        cutoff,
        aggr="add",
        **kwargs
    ):
        super(ConvCF, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.cutoff = cutoff

        self.input = nn.Linear(in_channels, num_filters)
        self.nn = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            # ShiftedSoftplus(),
            # nn.Linear(num_filters, num_filters),
        )
        self.output = nn.Linear(num_filters, out_channels)

    def forward(self, x, edge_index, edge_attr, gauss_exp):
        C = 0.5 * (torch.cos(edge_attr * torch.pi / self.cutoff) + 1)
        W = self.nn(gauss_exp) * C.view(-1, 1)

        x = self.input(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.output(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(self, start=0, stop=5, num_gaussians=50, **kwargs):
        super(GaussianSmearing, self).__init__(**kwargs)
        offsets = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offsets[1] - offsets[0]) ** 2
        self.register_buffer("offsets", offsets)

    def forward(self, dist):
        gaussian_exp = (dist[:] - self.offsets[None, :]) ** 2
        return torch.exp(self.coeff * gaussian_exp)


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class SchNetEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=5,
        num_gaussians=50,
        cutoff=4.5,
    ):
        super(SchNetEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff

        self.embedding = nn.Embedding(100, hidden_channels, padding_idx=0)

        self.gaussian_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactons = nn.ModuleList()
        for _ in range(num_interactions):
            self.interactons.append(
                InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            )

    def forward(self, batch):
        x = self.embedding(batch.atomic_numbers.int())
        gauss_exp = self.gaussian_expansion(batch.edge_attr)

        for interaction in self.interactons:
            x = x + interaction(x, batch.edge_index, batch.edge_attr, gauss_exp)

        batch.x = x
        return batch


"""
PaiNN Encoder

paper: https://arxiv.org/abs/2102.03150

Non-finished!
"""

class PaiNNConvCF(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_filters,
        num_gaussians,
        cutoff,
        aggr="add",
        **kwargs
    ):
        super(ConvCF, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.cutoff = cutoff

        self.input = nn.Linear(in_channels, num_filters)
        self.nn = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            
        )
        self.output = nn.Linear(num_filters, out_channels)

    def forward(self, h, x, edge_index, edge_attr, gauss_exp):
        C = 0.5 * (torch.cos(edge_attr * torch.pi / self.cutoff) + 1)
        W = self.nn(gauss_exp) * C.view(-1, 1)

        h = self.input(h)
        h = self.propagate(edge_index, h=h, x=x, W=W)
        h = self.output(h)
        return h, x

    def message(self, h_j, W):
        return h_j * W
    
class PaiNNInteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_gaussians = num_gaussians
        self.num_filters = num_filters
        self.cutoff = cutoff

        self.conv = PaiNNConvCF(
            hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff
        )
        self.act = nn.SiLU()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, h, x, edge_index, edge_attr, gauss_exp):
        h, x = self.conv(h, x, edge_index, edge_attr, gauss_exp)
        h = self.act(h)
        h = self.lin(h)
        return h, x

class PaiNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=5,
        num_gaussians=50,
        cutoff=4.5,
    ):
        super(SchNetEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff

        self.scalar_embedding = nn.Embedding(100, hidden_channels, padding_idx=0)

        self.gaussian_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            self.interactoins.append(
                PaiNNInteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            )
            

    def forward(self, batch):
        h = self.embedding(batch.atomic_numbers.int())
        x = torch.zeros((batch.num_nodes, self.hidden_channels))
        
        gauss_exp = self.gaussian_expansion(batch.edge_attr)

        for interaction in self.interactions:
            x = x + interaction(h, x, batch.edge_index, batch.edge_attr, gauss_exp)

        batch.x = x
        return batch
    

    
"""
E(3)-GNN
__________________

Paper: https://proceedings.mlr.press/v139/satorras21a.html

"""


class ConvEGNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, aggr="add", **kwargs):
        super(ConvEGNN, self).__init__(aggr=aggr, **kwargs)
        self.hidden_channels = hidden_channels

        self.phi_e = nn.Sequential(
            nn.Linear(in_channels * 2 + 1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
        )
        self.phi_inf = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, h, x, edge_index, edge_attr):
        x_diff = x[edge_index[0]] - x[edge_index[1]]
        r_ij = torch.norm(x_diff, dim=-1, keepdim=True)
        h, x = self.propagate(
            edge_index, x=x, h=h, edge_attr=edge_attr, x_diff=x_diff, r_ij=r_ij
        )
        return h, x

    def message(self, h_i, h_j, edge_attr, r_ij):
        m_ij = self.phi_e(torch.cat([h_i, h_j, r_ij], dim=-1))  # eq. (3)
        return m_ij

    def propagate(self, edge_index, size=None, *args, **kwargs):
        size = self._check_input(edge_index, size=None)        
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", coll_dict)
        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
        update_kwargs = self.inspector.distribute("update", coll_dict)

        m_ij = self.message(**msg_kwargs)

        W_ij = self.phi_x(m_ij)  # eq. (4)
        x = kwargs["x"]
        x += self.aggregate(W_ij * kwargs["x_diff"], **aggr_kwargs)  # eq. (4)

        m_i = self.aggregate(m_ij , **aggr_kwargs)  # eq. (5)/(8) # * self.phi_inf(m_ij)

        h = kwargs["h"]
        h += self.phi_h(torch.cat([kwargs["h"], m_i], dim=-1))  # eq. (6)

        return self.update((h, x), **update_kwargs)


class EGGNEncoder(nn.Module):
    def __init__(self, hidden_channels, n_layers=2):
        super(EGGNEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.embedding = nn.Embedding(100, hidden_channels, padding_idx=0)

        self.conv = nn.ModuleList()
        for _ in range(n_layers):
            self.conv.append(ConvEGNN(hidden_channels, hidden_channels))

    def forward(self, batch):
        h = self.embedding(batch.atomic_numbers.int())
        x = batch.pos.clone()

        for conv in self.conv:
            h, x = conv(h, x, batch.edge_index, batch.edge_attr)

        batch.x = x
        batch.h = h
        return batch


"""
Energy-head model
"""


class EnergyHead(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=5):
        super(EnergyHead, self).__init__()
        self.hidden_dim = hidden_dim
        # self.encoder = GNNEncoder(hidden_dim=hidden_dim, n_layers=n_layers)
        # self.encoder = SchNetEncoder(
        #     hidden_channels=hidden_dim, num_interactions=n_layers
        # )
        self.encoder = EGGNEncoder(hidden_channels=hidden_dim, n_layers=n_layers)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        batch = self.encoder(batch)
        y = self.out(batch.h).sum().view(-1)
        return y


"""
Training method
"""


def train(train_loader, epochs=100, lr=1e-3, load=False):
    model = EnergyHead(hidden_dim=128, n_layers=5)
    if load:
        model.load_state_dict(torch.load("egnn_model.pth"))
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = 1e10
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out, batch.y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(
            "Epoch: {:03d}, Loss: {:.7f}".format(epoch, epoch_loss / len(train_loader))
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = model.state_dict()

    torch.save(best_model, "egnn_model.pth")
    print("Best loss: {:.7f}".format(best_loss / len(train_loader)))
    model.load_state_dict(best_model)
    return model


if __name__ == "__main__":
    from data import (
        get_graph_dataloader,
        get_test_graphs,
        get_test_rotation_graphs,
        get_test_translation_graphs,
        get_test_permutation_graphs,
    )

    data = get_graph_dataloader()
    batch = next(iter(data))
    print(batch)
    # encoder = GNNEncoder()
    # encoder = SchNetEncoder()
    encoder = EGGNEncoder(hidden_channels=128, n_layers=2)
    print(encoder)
    out = encoder(batch)
    print(out)

    model = train(data, epochs=1000, lr=1e-4, load=True)

    print('-'*10, 'Rotation Test', '-'*10)
    graphs = get_test_rotation_graphs(2)
    g0, g1 = graphs[0], graphs[1]
    model(g0)
    model(g1)
    rotation_angle = 45
    import numpy as np
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                  [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                  [0, 0, 1]])    
    print('Rotated after encoding:\n', np.dot(g0.x.detach().numpy(), R))
    print('Structure rotated before encoding:\n', g1.x.detach().numpy())

    print(g0.h[:,:5])
    print(g1.h[:,:5])

    print('-'*10, 'Translation Test', '-'*10)
    graphs = get_test_translation_graphs(2)
    g0, g1 = graphs[0], graphs[1]
    model(g0)
    model(g1)
    l = 1
    print('Translated after encoding:\n', g0.x.detach().numpy() + l)
    print('Structure translated before encoding:\n', g1.x.detach().numpy())

    print(g0.h[:,:5])
    print(g1.h[:,:5])

    print('-'*10, 'Permutation Test', '-'*10)
    graphs = get_test_permutation_graphs(2)
    g0, g1 = graphs[0], graphs[1]
    model(g0)
    model(g1)
    print('Permuted after encoding:\n', g0.x.detach().numpy()[[1, 0]])
    print('Structure permuted before encoding:\n', g1.x.detach().numpy())

    print(g0.h[:,:5])
    print(g1.h[:,:5])
    

    
    graphs = get_test_graphs(100)    

    y = []
    y_hat = []
    with torch.no_grad():
        for graph in graphs:
            y.append(graph.y.item())
            y_hat.append(model(graph).item())
        loss = F.mse_loss(torch.tensor(y_hat), torch.tensor(y))
        print("Test Loss: {:.7f}".format(loss.item()))

    import matplotlib.pyplot as plt

    plt.plot(y, label="y")
    plt.plot(y_hat, label="y_hat")
    plt.legend()
    plt.show()
