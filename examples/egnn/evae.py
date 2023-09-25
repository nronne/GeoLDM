import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.distributions import Normal
from copy import deepcopy


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

        m_i = self.aggregate(m_ij, **aggr_kwargs)  # eq. (5)/(8) # * self.phi_inf(m_ij)

        h = kwargs["h"]
        h += self.phi_h(torch.cat([kwargs["h"], m_i], dim=-1))  # eq. (6)

        return self.update((h, x), **update_kwargs)


class EGGNEncoder(nn.Module):
    def __init__(self, hidden_channels, n_layers=2, embed=True):
        super(EGGNEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.embed = embed

        self.embedding = nn.Embedding(100, hidden_channels, padding_idx=0)

        self.conv = nn.ModuleList()
        for _ in range(n_layers):
            self.conv.append(ConvEGNN(hidden_channels, hidden_channels))

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
        zx,zh = batch_hat.x, batch_hat.h
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
        
        return mse_loss + kl_weight*kl_loss


def train(model, data, optimizer, epochs=100):
    model.train()
    best_loss = 1e10
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in data:
            batch = batch.clone()
            optimizer.zero_grad()
            loss = model.loss(batch, kl_weight=5.0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / batch.num_graphs
        print(f"Epoch {epoch} Loss: {epoch_loss / len(data)}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_params = deepcopy(model.state_dict())

    # freeze the encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    best_loss = 1e10
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in data:
            batch = batch.clone()
            optimizer.zero_grad()
            loss = model.loss(batch, kl_weight=0.0)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / batch.num_graphs
        print(f"Epoch {epoch} Loss: {epoch_loss / len(data)}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_params = deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    print('Best loss: ', best_loss / len(data))
    return model
        


if __name__ == "__main__":
    from data import (
        get_graph_dataloader,
    )
    from ase import Atoms
    from ase.io import write

    encoder = EGGNEncoder(hidden_channels=128, n_layers=3)
    decoder = EGGNEncoder(hidden_channels=128, n_layers=3, embed=False)

    model = AE(encoder, decoder)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    data = get_graph_dataloader(200, batch_size=32)
    # batch = data.dataset[-1]
    batch = next(iter(data))
    print(batch)

    # Train and save model:
    # model = train(model, data, optimizer, 2500)
    # torch.save(model.state_dict(), "best-ae2-model.pth")

    # Load model:
    model.load_state_dict(torch.load("best-ae2-model.pth"))

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
        for batch in get_graph_dataloader():
            for graph in batch.to_data_list():
                graph = graph.clone()
                # batch_hat, zx, zh = model(graph.clone())
                batch_z = model.encoder(graph.clone())
                zx, zh = batch_z.x, batch_z.h

                #add a bit of noise to the zs (5%)
                # zx += torch.randn_like(zx) * 0.05
                zh += torch.randn_like(zh) * 0.05

                batch.pos = zx
                batch.h = zh

                batch_hat = model.decoder(batch_z.clone())
                
                N = zx.shape[0]
                data.append(Atoms(f'C{N}', positions=batch_hat.x.detach().numpy(), cell=[5]*3))
                z_data.append(Atoms(f'C{N}', positions=zx, cell=[5]*3))
                true_data.append(Atoms(f'C{N}', positions=graph.pos.detach().numpy(), cell=[5]*3))
        write('t-data-hat.traj', data)
        write('t-data-z.traj', z_data)
        write('t-data-true.traj', true_data)
    
