import os
import torch
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.visualize import view
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph, Distance
from ase.optimize import BFGS

class Converter:
    def __init__(self, radius: float = 4.5):
        self.radius = radius
        # this needs to be changed to include the periodic boundary conditions at some point
        self.rg = RadiusGraph(r=radius, loop=True, max_num_neighbors=200)
        self.dt = Distance(norm=False)

    def convert(self, atoms: Atoms):
        """Convert a single atomic stucture to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

            sid (uniquely identifying object): An identifier that can be used to track the structure in downstream
            tasks. Common sids used in OCP datasets include unique strings or integers.

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with positions, atomic_numbers, tags,
            and optionally, energy, forces, distances, edges, and periodic boundary conditions.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        # set the atomic numbers, positions, and cell
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
        positions = torch.Tensor(atoms.get_positions())
        cell = torch.Tensor(np.array(atoms.get_cell())).view(1, 3, 3)
        natoms = positions.shape[0]
        y = torch.Tensor([atoms.get_potential_energy()])

        # put the minimum data in torch geometric data object
        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            y=y,
        )

        self.rg(data)
        self.dt(data)

        return data

    def convert_all(self, atoms_list: list):
        """Convert a list of atomic stuctures to a list of graphs.

        Args:
            atoms_list (list): A list of ASE atoms objects.

        Returns:
            data_list (list): A list of torch geometic data objects.
        """
        data_list = []
        for atoms in atoms_list:
            data = self.convert(atoms)
            data_list.append(data)
        return data_list


def get_atoms_data(N):
    data = read("C30-dataset.traj", index=":")
    data = []
    # for r in np.linspace(0.95, 3, N):
    #     cell = np.diag([10, 10, 10])
    #     x0 = np.random.uniform(0, 10, size=(3,))
    #     vec = np.random.uniform(-1, 1, size=(3,))
    #     vec = vec / np.linalg.norm(vec)
    #     x1 = x0 + r * vec
    #     atoms = Atoms('C2', positions=[x0, x1], cell=cell)
    #     atoms.calc = LennardJones()
    #     E = atoms.get_potential_energy()
    #     data.append(atoms)
    # for _ in range(N):
    #     n = np.random.randint(2, 8)
    #     cell = np.diag([4] * 3)
    #     positions = np.random.uniform(0, 4, size=(n, 3))
    #     atoms = Atoms('C' + str(n), positions=positions, cell=cell)
    #     atoms.calc = LennardJones()
    #     dyn = BFGS(atoms)
    #     dyn.run(fmax=0.05)
    #     atoms.center()
    #     data.append(atoms)
    #     E = atoms.get_potential_energy()
        
    # write('lj-data.traj', data)
    data = read("lj-data.traj", index=f":{N}")
    return data
    
def get_graph_data(N=30):

    data = get_atoms_data(N)


    a2g = Converter(radius=4.5)
    graphs = a2g.convert_all(data)
    
    return graphs

def get_test_rotation_graphs(N=30):
    data = read("C30-dataset.traj", index=":")
    data = []
    for r in np.linspace(0.95, 3, N):
        atoms = Atoms('C2', positions=[[0, 0, 0], [r, 0, 0]], cell=np.diag([10, 10, 10]))
        atoms.calc = LennardJones()
        E = atoms.get_potential_energy()
        data.append(atoms)

        rotation_angle = 45
        R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                    [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                    [0, 0, 1]])
        positions = np.array([[0, 0, 0], [r, 0, 0]])
        positions = np.dot(positions, R)
        atoms = Atoms('C2', positions=positions, cell=np.diag([10, 10, 10]))
        atoms.calc = LennardJones()
        E = atoms.get_potential_energy()
        data.append(atoms)

    a2g = Converter(radius=4.5)
    graphs = a2g.convert_all(data)
    
    return graphs

def get_test_translation_graphs(N=30):
    data = read("C30-dataset.traj", index=":")
    data = []
    for r in np.linspace(0.95, 3, N):
        atoms = Atoms('C2', positions=[[0, 0, 0], [r, 0, 0]], cell=np.diag([10, 10, 10]))
        atoms.calc = LennardJones()
        E = atoms.get_potential_energy()
        data.append(atoms)

        l = 1
        positions = np.array([[0, 0, 0], [r, 0, 0]])
        positions = positions + l
        atoms = Atoms('C2', positions=positions, cell=np.diag([10, 10, 10]))
        atoms.calc = LennardJones()
        E = atoms.get_potential_energy()
        data.append(atoms)

    a2g = Converter(radius=4.5)
    graphs = a2g.convert_all(data)
    
    return graphs        

def get_test_permutation_graphs(N=30):
    data = read("C30-dataset.traj", index=":")
    data = []
    for r in np.linspace(0.95, 3, N):
        atoms = Atoms('C2', positions=[[0, 0, 0], [r, 0, 0]], cell=np.diag([10, 10, 10]))
        atoms.calc = LennardJones()
        E = atoms.get_potential_energy()
        data.append(atoms)

        positions = np.array([[0, 0, 0], [r, 0, 0]])
        positions = positions[::-1]
        atoms = Atoms('C2', positions=positions, cell=np.diag([10, 10, 10]))
        atoms.calc = LennardJones()
        E = atoms.get_potential_energy()
        data.append(atoms)

    a2g = Converter(radius=4.5)
    graphs = a2g.convert_all(data)
    
    return graphs        


def get_test_graphs(N=30):
    data = read("C30-dataset.traj", index=":")
    data = []
    for r in np.linspace(0.95, 3, N):
        atoms = Atoms('C2', positions=[[0, 0, 0], [r, 0, 0]], cell=np.diag([10, 10, 10]))
        atoms.calc = LennardJones()
        E = atoms.get_potential_energy()
        data.append(atoms)
    
    a2g = Converter(radius=4.5)
    graphs = a2g.convert_all(data)
    
    return graphs
    

def get_graph_dataloader(N=30, batch_size=32):
    graphs = get_graph_data(N=N)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    return loader

if (__name__=="__main__"):
    dataloader = get_graph_dataloader()
    example = next(iter(dataloader))
    print(example.pos)
