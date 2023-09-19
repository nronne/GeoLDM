import torch
import torch.nn as nn
from egnn.egnn_new import EGNN


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        context_dim,
        hidden_dim,
        latent_dim,
        num_layers=3,
        equivariant_block_layers=2,            
        activation_function=nn.SiLU(),
        attention=False,
        device=torch.device("cpu"),
    ):
        super(Encoder, self).__init__()
        self.egnn = EGNN(
            in_node_nf=input_dim+context_dim,
            out_node_nf=hidden_dim,
            in_edge_nf=1,
            hidden_nf=hidden_dim,
            device=device,
            act_fn=activation_function,
            n_layers=num_layers,
            attention=attention,
            tanh=False,
            norm_constant=0.,
            inv_sublayers=equivariant_block_layers,
            sin_embedding=False,
            normalization_factor=1.0,
            aggregation_method="sum",
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation_function,
            nn.Linear(hidden_dim, latent_dim * 2 + 1))



