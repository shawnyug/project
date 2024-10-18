import scanpy as sc
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from itertools import cycle, product
from scipy import stats

import multiprocessing


def process_data(adata):
    # get the total counts per cell/spot
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    # the library size factor is defined as the total counts per cell/spot divided by the median total counts of all the cells. in order to keep all the cells/spots having the same number of counts
    adata.obs["size_factor"] = adata.obs["total_counts"] / np.median(adata.obs["total_counts"])
    #adata.obs["size_factor"] = adata.obs["total_counts"] / 10000
    adata.layers["raw_counts"] = adata.X
    adata.X = np.matmul(np.linalg.inv(np.diag(adata.obs["size_factor"])), adata.X)

    # log and calculate the z-score of the counts
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

def get_kendall_tau(tuple):
    x,y = tuple
    res = stats.kendalltau(x,y)
    return res.statistic

def get_spearmanr(tuple):
    x,y = tuple
    res = stats.spearmanr(x,y)
    return res.statistic

def calculate_rank_corr(adata1, adata2, gene_names, flag="Spearman"):
    mat1 = adata1[:,gene_names].X
    mat2 = adata2[:, gene_names].X

    # Get all combinations of two matrix columns
    combinations = list(product(mat1, mat2))
    pool = multiprocessing.Pool() #the cores equal to the number of cores of the machine
    if flag=="Spearman":
        fun = get_spearmanr
    elif flag=="kendalltau":
        fun = get_kendall_tau
    else:
        raise ValueError("Please check the rank correlation you want to compute")
    res = pool.map(fun, combinations)

    res = np.array(res).reshape((adata1.shape[0], adata2.shape[0]))

    return res

def calculate_corr(adata1, adata2, gene_names, flag="Pearson"):

    if flag=="Pearson":
        seq_data_copy = adata1.copy()
        spatial_data_partial_copy = adata2.copy()

        process_data(seq_data_copy)
        process_data(spatial_data_partial_copy)

        corr_sc_st = np.corrcoef(seq_data_copy[:, gene_names].X, spatial_data_partial_copy[:, gene_names].X, rowvar=True)
        corr_cross = corr_sc_st[:seq_data_copy.shape[0], seq_data_copy.shape[0]:]

        #corr_cross[corr_cross < 0] = 0

    else:
        corr_cross = calculate_rank_corr(adata1, adata2, gene_names, flag=flag)

    return corr_cross

def get_sample_weights(corr_matrix, topK=50, axis=None):
    # calculate the sample weights for sc and ST
    # corr_matrix shape: [cells, spots]
    # axis=0, weights for cells, find the most similar topK cells for each spot, then the occurrence frequency server as the probability
    ind_map = np.argsort(corr_matrix, axis=axis)
    if axis==0:
        topK_ind = ind_map[-topK:, :]
        df = pd.DataFrame(topK_ind.flatten()).value_counts(normalize=True)
    elif axis==1:
        topK_ind = ind_map[:, -topK:]
        df = pd.DataFrame(topK_ind.flatten()).value_counts(normalize=True)
    else:
        raise ValueError("user need to specify a axis")
    weights = np.zeros((corr_matrix.shape[axis]))
    ind = [i[0] for i in df.index.to_list()]
    weights[ind] = df.values

    return weights, topK_ind

import torch.nn.functional as F


# class TrainDL(DataLoader):
#     """Train data loader."""
#
#     def __init__(self, data_loader_dict, **kwargs):
#         self.data_loader_dict = data_loader_dict
#         idx_list = []
#         max_dl = {}
#         for mode, dl_list in data_loader_dict.items():
#             idx = np.argmax([dl.dataset.n_obs for dl in dl_list])
#             idx_list.append(idx)
#             max_dl[mode] = dl_list[idx]
#         self.largest_train_dl_idx = idx_list
#         self.largest_dl = self.data_loader_dict["sc"][0]
#         super().__init__(self.largest_dl, **kwargs)
#
#     def __len__(self):
#         return len(self.largest_dl)
#
#     def __iter__(self):
#         train_dls_list = []
#         for largest_idx, (mode, dl_list) in zip(self.largest_train_dl_idx, self.data_loader_dict.items()):
#             train_dls = [
#                 dl if i == largest_idx else cycle(dl)
#                 for i, dl in enumerate(dl_list)
#             ]
#             train_dls_list.extend(train_dls)
#             #train_dls_list[mode] = train_dls
#         return zip(*train_dls_list)

class TrainDL(DataLoader):
    """Train data loader."""

    def __init__(self, data_loader_list, **kwargs):
        self.data_loader_list = data_loader_list
        self.largest_train_dl_idx = np.argmax(
            [dl.dataset.n_obs for dl in data_loader_list]
        )
        self.largest_dl = self.data_loader_list[self.largest_train_dl_idx]
        super().__init__(self.largest_dl, **kwargs)

    def __len__(self):
        return len(self.largest_dl)

    def __iter__(self):
        train_dls = [
            dl if i == self.largest_train_dl_idx else cycle(dl)
            for i, dl in enumerate(self.data_loader_list)
        ]
        return zip(*train_dls)