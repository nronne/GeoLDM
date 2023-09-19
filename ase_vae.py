import torch
import utils
import numpy as np
from egnn.models import EGNN_encoder_QM9, EGNN_decoder_QM9
from equivariant_diffusion.en_diffusion import EnHierarchicalVAE
from qm9.data.dataset_class import ProcessedDataset
from torch.utils.data import DataLoader
from ase.io import read, write

from equivariant_diffusion.utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    assert_correctly_masked,
    sample_center_gravity_zero_gaussian_with_mask,
)
from torch.distributions.categorical import Categorical
from ase import Atoms


def dict_to_atoms(d):
    num_atoms = d["num_atoms"]
    a = Atoms(f"X{num_atoms}")
    a.set_atomic_numbers(d["charges"][:num_atoms])
    a.set_positions(d["positions"][:num_atoms, :])
    return a


class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


def traj_to_dict(traj):
    max_n_atoms = max([len(a) for a in traj])
    d = {
        "num_atoms": [],
        "charges": [],
        "positions": [],
    }
    for atoms in traj:
        d["num_atoms"].append(len(atoms))
        charges = torch.tensor(atoms.get_atomic_numbers())
        # pad charges with zeros
        charges = torch.cat([charges, torch.zeros(max_n_atoms - len(charges))])
        d["charges"].append(charges)

        positions = torch.tensor(atoms.get_positions())
        # pad positions with zeros
        positions = torch.cat(
            [positions, torch.zeros((max_n_atoms - len(positions), 3))]
        )
        d["positions"].append(positions)

    d["num_atoms"] = torch.tensor(d["num_atoms"])
    d["charges"] = torch.stack(d["charges"])
    d["positions"] = torch.stack(d["positions"])
    return d


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


def collate_fn(batch):
    """
    Collation function that collates datapoints into the batch format for cormorant

    Parameters
    ----------
    batch : list of datapoints
        The data to be collated.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """
    batch = {
        prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()
    }

    to_keep = batch["charges"].sum(0) > 0

    batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

    atom_mask = batch["charges"] > 0
    batch["atom_mask"] = atom_mask

    # Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    # edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    batch["edge_mask"] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    batch["charges"] = batch["charges"].unsqueeze(2)

    return batch


def count_n_nodes(data):
    n_nodes = {}
    for d in data:
        n = len(d)
        if n not in n_nodes:
            n_nodes[n] = 1
        else:
            n_nodes[n] += 1
    return n_nodes


def encoder_decoder(data):
    all_symbols = []
    for d in data:
        all_symbols.extend(d.get_chemical_symbols())
    decoder = np.unique(all_symbols)
    
    species = []
    for symbol in decoder:
        a = Atoms(symbol)
        species.append(a.get_atomic_numbers()[0])
    idx = np.argsort(species)
    
    decoder = decoder[idx]
    encoder = {s: i for i, s in enumerate(decoder)}
    species = np.array(species).flatten()[idx]
    return encoder, decoder, torch.tensor(species)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

data = read("qm9/temp/qm9/train.traj", ":18")
test = read("qm9/temp/qm9/train.traj", "18:")


encoder, decoder, species = encoder_decoder(data)
print("encoder", encoder)
print("decoder", decoder)

data_dict = traj_to_dict(data)

dataset = ProcessedDataset(data_dict, subtract_thermo=False)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
)

test_data_dict = traj_to_dict(test)
test_dataset = ProcessedDataset(test_data_dict, subtract_thermo=False)
test_dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
)

data_dummy = next(iter(dataloader))
# print("data_dummy", data_dummy)  # dummy data - not sure how much of it is actually used

n_nodes = count_n_nodes(data)

dataset_info = {"atom_decoder": encoder, "atom_encoder": decoder, "n_nodes": n_nodes}
nodes_dist = DistributionNodes(dataset_info["n_nodes"])

include_charges = True
in_node_nf = len(dataset_info["atom_decoder"]) + int(include_charges)
context_node_nf = 0
latent_nf = 4  # this is the size of the scalar latent representation
nf = 128

encoder = EGNN_encoder_QM9(
    in_node_nf=in_node_nf,
    context_node_nf=context_node_nf,
    out_node_nf=latent_nf,
    n_dims=3, # this is the E(n) equivariant dimension (ie. 3 for 3D space)
    device=device,
    hidden_nf=nf,
    act_fn=torch.nn.SiLU(),
    n_layers=1,
    attention=True,
    tanh=True,
    mode="egnn_dynamics",
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=False,
    normalization_factor=1.0,
    aggregation_method="sum",
    include_charges=include_charges,
)

decoder = EGNN_decoder_QM9(
    in_node_nf=latent_nf,
    context_node_nf=context_node_nf,
    out_node_nf=in_node_nf,
    n_dims=3,
    device=device,
    hidden_nf=nf,
    act_fn=torch.nn.SiLU(),
    n_layers=6,
    attention=True,
    tanh=True,
    mode="egnn_dynamics",
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=False,
    normalization_factor=1.0,
    aggregation_method="sum",
    include_charges=include_charges,
)

vae = EnHierarchicalVAE(
    encoder=encoder,
    decoder=decoder,
    in_node_nf=in_node_nf,
    n_dims=3,
    latent_node_nf=latent_nf,
    kl_weight=0.1,
    norm_values=[1, 4, 1],
    include_charges=include_charges,
)


optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)


def compute_loss_and_nll(
    generative_model, nodes_dist, x, h, node_mask, edge_mask, context
):
    bs, n_nodes, n_dims = x.size()

    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

    assert_correctly_masked(x, node_mask)

    # Here x is a position tensor, and h is a dictionary with keys
    # 'categorical' and 'integer'.
    nll = generative_model(x, h, node_mask, edge_mask, context)

    N = node_mask.squeeze(2).sum(1).long()

    log_pN = nodes_dist.log_prob(N)

    assert nll.size() == log_pN.size()
    nll = nll - log_pN

    # Average over batch.
    nll = nll.mean(0)

    reg_term = torch.tensor([0.0]).to(nll.device)
    mean_abs_z = 0.0

    return nll, reg_term, mean_abs_z


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def train(
    model,
    optimizer,
    dataloader,
    n_epochs=1,
    augment_noise=0,
    ode_regularization=1e-3,
    n_report_steps=10,
    clip_grad=True,
    device="cpu",
):
    model.train()
    for epoch in range(n_epochs):
        nll_epoch = []
        for i, batch in enumerate(dataloader):
            x = batch["positions"].to(device, dtype)
            node_mask = batch["atom_mask"].to(device, dtype).unsqueeze(2)
            edge_mask = batch["edge_mask"].to(device, dtype)
            one_hot = batch["one_hot"].to(device, dtype)
            charges = batch["charges"].to(device, dtype)

            x = remove_mean_with_mask(x, node_mask)

            if augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(
                    x.size(), x.device, node_mask
                )
                x = x + eps * augment_noise

            x = remove_mean_with_mask(x, node_mask)

            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {"categorical": one_hot, "integer": charges}
            context = None

            optimizer.zero_grad()

            nll, reg_term, mean_abs_z = compute_loss_and_nll(
                model, nodes_dist, x, h, node_mask, edge_mask, context
            )
            loss = nll + ode_regularization * reg_term
            loss.backward()

            if clip_grad:
                grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            else:
                grad_norm = 0.0

            optimizer.step()

            # if i % n_report_steps == 0:
            #     print(
            #         f"\rEpoch: {epoch}, iter: {i}/{len(dataloader)}, "
            #         f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
            #         f"RegTerm: {reg_term.item():.1f}, "
            #         # f"GradNorm: {grad_norm:.1f}"
            #     )

            nll_epoch.append(nll.item())

        print(f"Epoch: {epoch}, NLL: {np.mean(nll_epoch):.2f}")


def test(model, dataloader, device="cpu"):
    model.eval()
    with torch.no_grad():
        nll_epoch = []
        batch = next(iter(dataloader))

        x = batch["positions"].to(device, dtype)
        node_mask = batch["atom_mask"].to(device, dtype).unsqueeze(2)
        edge_mask = batch["edge_mask"].to(device, dtype)
        one_hot = batch["one_hot"].to(device, dtype)
        charges = batch["charges"].to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)
        print("Charges:", charges.squeeze(0, 2))
        true_atom_dict = {
            "num_atoms": int(x.shape[1]),
            "charges": charges.squeeze(0, 2),
            "positions": x.squeeze(0),
        }

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {"categorical": one_hot, "integer": charges}
        context = None

        z_x_mu, sigma_0_x, z_h_mu, sigma_0_h = model.encode(
            x, h, node_mask, edge_mask, context
        )
        print("z_x_mu:", z_x_mu.shape)
        print("z_h_mu:", z_h_mu.shape)

        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        z_xh_sigma = torch.zeros_like(z_xh_mean)  #
        z_xh_sigma = torch.ones_like(z_xh_mean) * 0.01
        z_xh = model.sample_normal(z_xh_mean, z_xh_sigma, node_mask)

        print("latent shape:", z_xh.shape)

        x_recon, h_recon = model.decoder._forward(z_xh, node_mask, edge_mask, context)

        print(
            "h_recon:", torch.sigmoid(h_recon[:, :, :-1])
        )  # something with one-hot atom-types
        Z = np.round(h_recon[:, :, -1].squeeze(0).detach().cpu().numpy())
        recon_atom_dict = {
            "num_atoms": int(x_recon.shape[1]),
            "charges": Z,  # charges.squeeze(0,2),
            "positions": x_recon.squeeze(0),
        }
        write(
            "true-recon.traj",
            [dict_to_atoms(true_atom_dict), dict_to_atoms(recon_atom_dict)],
        )


train(vae, optimizer, dataloader, n_epochs=50)

test(vae, test_dataloader)
