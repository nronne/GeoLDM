import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.distributions import Normal
from copy import deepcopy


class ConvEGNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, aggr="add", use_edge_attr=True, **kwargs):
        super(ConvEGNN, self).__init__(aggr=aggr, **kwargs)
        self.hidden_channels = hidden_channels
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            n = 2
        else:
            n = 1

        self.phi_e = nn.Sequential(
            nn.Linear(in_channels * 2 + n, hidden_channels),
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
        if self.use_edge_attr:
            m_ij = self.phi_e(torch.cat([h_i, h_j, r_ij, edge_attr], dim=-1))  # eq. (3) (with edge_attr being the bond distance)
        else:
            m_ij = self.phi_e(torch.cat([h_i, h_j, r_ij], dim=-1))
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

        m_i = self.aggregate(m_ij * self.phi_inf(m_ij), **aggr_kwargs)  # eq. (5)/(8) #

        h = kwargs["h"]
        h += self.phi_h(torch.cat([kwargs["h"], m_i], dim=-1))  # eq. (6)

        return self.update((h, x), **update_kwargs)


class EGGNEncoder(nn.Module):
    def __init__(self, hidden_channels, n_layers=2, embed=True, use_edge_attr=True, **kwargs):
        super(EGGNEncoder, self).__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.embed = embed

        self.embedding = nn.Embedding(100, hidden_channels, padding_idx=0)

        self.conv = nn.ModuleList()
        for _ in range(n_layers):
            self.conv.append(ConvEGNN(hidden_channels, hidden_channels, use_edge_attr=use_edge_attr))

    def forward(self, batch):
        if self.embed:
            h = self.embedding(batch.atomic_numbers.int())
        else:
            h = batch.h.clone()

        x = batch.pos.clone()

        for conv in self.conv:
            h, x = conv(h, x, batch.edge_index, batch.edge_attr)

        batch.x = x
        batch.h = h
        return batch


class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        batch_hat = self.encoder(batch.clone())
        zx, zh = batch_hat.x, batch_hat.h
        batch.pos = zx
        batch.h = zh
        batch = self.decoder(batch)
        return batch, zx, zh

    def loss(self, batch, kl_weight=1.0):
        pos = batch.pos.clone()
        batch, zx, zh = self.forward(batch)
        mse_loss = F.mse_loss(batch.x.view(-1), pos.view(-1))

        z_mean = zx
        z_log_var = torch.zeros_like(z_mean)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        return mse_loss + kl_weight * kl_loss


def train(model, data, val_data, optimizer, epochs0=100, epochs1=10):
    model.train()
    best_loss = 1e10
    for epoch in range(epochs0):
        epoch_loss = 0
        for batch in data:
            batch = batch.clone()
            optimizer.zero_grad()
            loss = model.loss(batch, kl_weight=1.0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / batch.num_graphs
        print(f"Epoch {epoch} Loss: {epoch_loss / len(data)}", end="")

        val_loss = 0
        with torch.no_grad():
            for batch in val_data:
                batch = batch.clone()
                loss = model.loss(batch, kl_weight=1.0)
                val_loss += loss.item() / batch.num_graphs

        print(f" Val Loss: {val_loss / len(val_data)}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_params = deepcopy(model.state_dict())

    # freeze the encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    best_loss = 1e10
    for epoch in range(epochs1):
        epoch_loss = 0
        for batch in data:
            batch = batch.clone()
            optimizer.zero_grad()
            loss = model.loss(batch, kl_weight=0.0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / batch.num_graphs
        print(f"Epoch {epoch} Loss: {epoch_loss / len(data)}", end="")

        val_loss = 0
        with torch.no_grad():
            for batch in val_data:
                batch = batch.clone()
                loss = model.loss(batch, kl_weight=0.0)
                val_loss += loss.item() / batch.num_graphs

        print(f" Val Loss: {val_loss / len(val_data)}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_params = deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    print("Best loss: ", best_loss / len(data))
    return model


if __name__ == "__main__":
    from data import (
        get_graph_dataloaders,
    )
    from ase import Atoms
    from ase.io import write

    n_hidden = 64
    encoder = EGGNEncoder(hidden_channels=n_hidden, n_layers=4)
    decoder = EGGNEncoder(hidden_channels=n_hidden, n_layers=4, embed=False, use_edge_attr=False)

    model = AE(encoder, decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_data, val_data, test_data = get_graph_dataloaders(400, batch_size=32)
    # batch = data.dataset[-1]
    batch = next(iter(val_data))
    print(batch)

    # Train and save model:
    # model = train(model, train_data, val_data, optimizer, 0, 1000)
    # torch.save(model.state_dict(), "best-ae2-phi_inf-3-model.pth")

    # # Load model:
    model.load_state_dict(torch.load("best-ae2-phi_inf-3-model.pth"))

    # with torch.no_grad():
    #     pos = batch.pos.clone()
    #     batch_hat, zx, zh = model(batch.clone())
    #     pos_hat = batch_hat.x

    # print('pos', pos)
    # print('z', zx)
    # print('pos_hat', pos_hat)

    with torch.no_grad():
        data = []
        z_data = []
        true_data = []
        reg_loss = 0
        for batch in test_data:
            for graph in batch.to_data_list():
                graph_hat, zx, zh = model(graph.clone())

                mse = F.mse_loss(graph.pos.view(-1), graph_hat.x.view(-1))
                N = graph.num_nodes
                data.append(
                    Atoms(f"C{N}", positions=graph_hat.x.detach().numpy(), cell=[5] * 3)
                )
                
                z_data.append(Atoms(f"C{N}", positions=zx, cell=[5] * 3))
                
                true_data.append(
                    Atoms(f"C{N}", positions=graph.pos.detach().numpy(), cell=[5] * 3)
                )
                reg_loss += mse.item()
                
        reg_loss /= len(test_data)
        print("Reg Loss: ", reg_loss)
                
                
        write("t-data-hat.traj", data)
        write("t-data-z.traj", z_data)
        write("t-data-true.traj", true_data)
