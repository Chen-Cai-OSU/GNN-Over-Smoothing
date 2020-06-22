# Created at 2020-06-22
# Summary: utils
import copy
import os.path as osp
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg as LA
import torch
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs
from torch_geometric.data.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian
from torch_geometric.utils.convert import from_networkx


def fix_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tonp(tsr):
    if isinstance(tsr, np.ndarray):
        return tsr
    elif isinstance(tsr, np.matrix):
        return np.array(tsr)
    elif isinstance(tsr, scipy.sparse.csc.csc_matrix):
        return np.array(tsr.todense())

    assert isinstance(tsr, torch.Tensor)
    tsr = tsr.cpu()
    assert isinstance(tsr, torch.Tensor)

    try:
        arr = tsr.numpy()
    except TypeError:
        arr = tsr.detach().to_dense().numpy()
    except:
        arr = tsr.detach().numpy()

    assert isinstance(arr, np.ndarray)
    return arr


def eig(L):
    L = tonp(L)
    try:
        assert np.allclose(L, L.T)
    except AssertionError:
        plt.imshow(L)  # f'L is not symmetric. Diff is {np.max(L- L.T)}'
        plt.colorbar()
        plt.title('L-L.T')
        plt.show()
        sys.exit('Not sym matrix')

    w, v = LA.eigh(L)
    return w, v


def get_laplacian_mat(edge_index, edge_weight, num_node, normalization='sym'):  # todo: change back
    """ return a laplacian (torch.sparse.tensor)"""
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                            normalization=normalization)  # see https://bit.ly/3c70FJK for format
    return torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([num_node, num_node]))


class low_eig:
    def __init__(self, g):
        assert isinstance(g, Data), 'Not pyG graph'
        self.laplacian = get_laplacian_mat(g.edge_index, g.edge_weight, g.num_nodes,
                                           normalization='sym')  # get_laplacian(g.edge_index, normalization='sym')

    def __sparse_tensor2_sparse_numpyarray(self, sparse_tensor):
        """
        :param sparse_tensor: a COO torch.sparse.FloatTensor
        :return: a scipy.sparse.coo_matrix
        """
        if sparse_tensor.device.type == 'cuda':
            sparse_tensor = sparse_tensor.to('cpu')

        values = sparse_tensor._values().numpy()
        indices = sparse_tensor._indices()
        rows, cols = indices[0, :].numpy(), indices[1, :].numpy()
        size = sparse_tensor.size()
        scipy_sparse_mat = coo_matrix((values, (rows, cols)), shape=size, dtype=np.float)
        return scipy_sparse_mat

    def get_eig(self, k, which='SR', laplacian=None, mode=None):
        """
        :param laplacian: The input laplacian matrix, should be a sparse tensor.
        :param N: the size of the graph.
        :param k: The top K (smalleset) eigenvectors.
        :param which: LM, SM, LR, SR, LM, SM largest/smallest magnitude, LR/SR largest/smallest real value.
        more details see scipy.sparse.linalg.eigs
        :return: return top K eigenvec. in the format of a N * k tensor. All vectors are automatically normalized.
        """
        if laplacian is None:
            laplacian = self.laplacian

        assert isinstance(laplacian, (torch.sparse.FloatTensor, torch.cuda.sparse.FloatTensor)), \
            f'input laplacian must be sparse tensor. Got {type(laplacian)}'
        # we need to convert the sparse tensor to scipy sparse mat, so that we can apply
        # the functions scipy.sparse.linalg.eigs() which should be faster than other methods.
        scipy_lap = self.__sparse_tensor2_sparse_numpyarray(laplacian)
        M, N = scipy_lap.shape
        assert (M == N and k < N - 1), \
            f'Input laplacian must be a square matrix. To use scipy method, {k} (#eigvecs) < {N - 1} (size of laplacian - 1).'

        if mode == None:
            vals, vecs = eigs(scipy_lap, k=k, which=which)
        else:
            vals, vecs = eig(scipy_lap.todense())
            vals, vecs = vals[:k], vecs[:, :k]

        vecs = torch.FloatTensor(vecs.real)
        vecs = tonp(vecs)
        # vecs = self.__normalize(vecs) # no effect
        return vecs

    def mix_low_eig(self, k=10, n_vec=30, mode=None):
        vecs = self.get_eig(k, mode=mode)
        coef = np.random.random((k, n_vec))
        mix_vec = np.dot(vecs, coef)
        return mix_vec


def energy(v1, L1):
    """ compute the energy
        v1: n * d
        L1 : n * n
        return tr(v.T * L * v)
    """

    L1 = tonp(L1)
    assert v1.shape[0] == L1.shape[0] == L1.shape[1]
    E = np.dot(np.dot(v1.T, L1), v1)
    E = np.diag(E)
    return E


def sample_index(n_edge):
    idx = np.random.choice(range(n_edge), 1)[0]
    if idx % 2 == 0:
        idx += 1
    return idx, idx - 1


class random_pyG():
    # todo: refactor
    def __init__(self):
        fix_seed()

    def reorder_edge_index(self, edge_index):
        """ for an edge index, reorder it so that each column
            [1, 2].T and [2,1].T are contiguious. A pretty useful subroutine.
        """
        edge_index = tonp(edge_index)
        n = edge_index.shape[1]
        assert edge_index.shape[0] == 2
        assert n % 2 == 0, 'Edge number should be even.'
        tmp = edge_index.T.tolist()
        tmp = [tuple(k) for k in tmp]

        edge2idx = dict(zip(tmp, range(n)))
        idx2edge = dict(zip(range(n), tmp))

        new_index = []
        for i in range(n):
            u, v = idx2edge[i]
            j = edge2idx[(v, u)]
            if (i not in new_index) and (j not in new_index):
                new_index.append(i)
                new_index.append(j)
        return new_index

    def reorder_pyG(self, g):
        new_index = self.reorder_edge_index(g.edge_index)

        new_edge_index = g.edge_index[:, new_index]
        new_edge_weight = g.edge_weight[new_index]

        new_g = Data(edge_index=new_edge_index, edge_weight=new_edge_weight)
        return new_g

    def process_nx_graph(self, g, add_weight=False, uniform_weight=False):
        """ given a nx graph. add random edge_weight"""

        if isinstance(g, Data):
            print('already a pygeo graph')
            g = self.reorder_pyG(g)
            return g

        if add_weight:
            for u, v in g.edges:
                g[u][v]['edge_weight'] = np.random.random()

        if uniform_weight:
            for u, v in g.edges:
                g[u][v]['edge_weight'] = 0.1

        g = from_networkx(g)
        g = self.reorder_pyG(g)
        return g

    def random_rm_edges(self, g, n=1):
        """given a nx graph, random remove some edges """
        g_copy = copy.deepcopy(g)
        edges = list(g.edges)
        assert n < len(edges)
        chosen_edges = random.sample(edges, k=n)
        # for edge in chosen_edges:
        #     g.remove_edge(edge[0], edge[1])
        # g = g.remove_edge(edge[1], edge[0])

        g_copy.remove_edges_from(chosen_edges)
        return g_copy

    def increase_random_edge_w(self, g, n=1, w=1000):
        """ randomly increase the weight of edge. change the weight to w """

        g = copy.deepcopy(g)
        n_edge = g.edge_index.size(1) // 2
        assert n < n_edge, f'n {n} has to be smaler than {n_edge}.'
        change_edges = random.sample(range(n_edge), k=n)
        new_weight = g.edge_weight
        for idx in change_edges:
            new_weight[2 * idx] = w
            new_weight[2 * idx + 1] = w

        new_index = g.edge_index
        new_g = Data(edge_index=new_index, edge_weight=new_weight)
        return new_g

    def rm_pyG_edges(self, g, n=1):
        n_edge = g.edge_index.size(1) // 2
        retain_edges = random.sample(range(n_edge), k=n_edge - n)
        indices = []
        for idx in retain_edges:
            indices.append(2 * idx)
            indices.append(2 * idx + 1)

        new_edge_index = g.edge_index[:, indices]
        new_edge_weight = g.edge_weight[indices]
        new_g = Data(edge_index=new_edge_index, edge_weight=new_edge_weight)
        return new_g

    def get_planetoid(self, dataset='cora'):
        path = osp.join('/home/cai.507/Documents/DeepLearning/sparsifier/sparsenet', 'data', dataset)
        dataset = Planetoid(path, dataset, T.TargetIndegree())
        n_edge = dataset.data.edge_index.size(1)
        g = Data(edge_index=dataset.data.edge_index, edge_weight=torch.ones(n_edge))
        assert g.is_directed() == False
        # g = g.coalesce()
        return g
