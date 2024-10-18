import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
from scipy.stats import spearmanr
import time
import parser
import argparse

import os
import sys
import pandas as pd
from anndata import AnnData
import anndata as ad
import scanpy as sc
import scenvi
from tqdm import tqdm

sys.path.append("../")
from imputeSTFormer import utils

parser = argparse.ArgumentParser(description='ENVI training')
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--data_path", type=str, default="./data/imputation/DataUpload", required=False)
parser.add_argument("--data_id", type=str, default="Dataset9", required=False)
parser.add_argument('--batch_size', default=128, type=int, help='number of batch_size')
parser.add_argument("--min_cells", type=int, default=1)
parser.add_argument("--min_genes", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out_file_path", type=str, default="../../results", required=False)
parser.add_argument("--epochs", type=int, default=16000)


if __name__ == "__main__":

    # load dataset and preprocess data
    args = parser.parse_args()

    if args.experiment_name:
        out_file_path = os.path.join(args.out_file_path, args.experiment_name)
        if not os.path.isdir(out_file_path):
            os.mkdir(out_file_path)
        args.out_file_path = out_file_path

    sc.settings.figdir = args.out_file_path
    sc.settings.set_figure_params(vector_friendly=True)
    # load & process data
    #seq_data, spatial_data = load_data()
    data_root = os.path.join("../../data", args.data_id)
    seq_data = sc.read_h5ad(f"{data_root}_sc.h5ad")
    spatial_data = sc.read_h5ad(f"{data_root}_st.h5ad")
    spatial_data.obsm['spatial'] = np.array(spatial_data.obs[['x', 'y']])

    train_size = 0.8
    spatial_data.var_names = [x.lower() for x in spatial_data.var_names]
    seq_data.var_names = [x.lower() for x in seq_data.var_names]

    spatial_data.var_names_make_unique()
    seq_data.var_names_make_unique()

    # subset spatial data into shared genes
    gene_names = np.intersect1d(spatial_data.var_names, seq_data.var_names)

    # only use genes in both datasets
    #seq_data = seq_data[:, gene_names].copy()
    spatial_data = spatial_data[:, gene_names].copy()

    seq_gene_names = seq_data.var_names
    n_genes = spatial_data.n_vars
    n_train_genes = int(n_genes * train_size)

    # set the seed for random and torch
    #seed_torch(args.seed)

    # randomly permute all the shared genes
    np.random.seed(seed=0)
    rand_gene_idx = np.random.choice(range(n_genes), n_genes, replace=False)

    fold = 5
    test_gene_list = []
    st_imputed = []

    start = time.time()
    for train_genes, test_genes in tqdm(
            utils.cv_data_gen(rand_gene_idx, kfold=fold), total=fold
    ):
        # spatial_data_partial has a subset of the genes to train on
        spatial_data_partial = spatial_data[:, train_genes].copy()

        # model training
        envi_model = scenvi.ENVI(spatial_data=spatial_data_partial, sc_data=seq_data,
                                 sc_genes=gene_names)  # specify all the genes in ST in case not in top 2048 HVG, and will be missing when performing 5-CV

        envi_model.train(epochs=args.epochs, batch_size=args.batch_size)
        envi_model.impute_genes()
        envi_model.infer_niche()

        test_gene_list.append(test_genes)

        test_imputed = np.array(envi_model.spatial_data.obsm['imputation'][gene_names[test_genes]])
        st_imputed.append(test_imputed)

    end = time.time()
    print('running time = {}'.format(end - start))
    # spatial_data_partial has a subset of the genes to train on
    test_gene_ind = np.concatenate(test_gene_list)

    spatial_data = spatial_data[:, test_gene_ind].copy()

    # spatial_data.layers["X_binary"] = np.apply_along_axis(binary, 0, spatial_data.copy().X)
    # spatial_data.obsm["zero_prob"] = nb_zero_prob
    spatial_data.obsm["imputed"] = np.hstack(st_imputed)

    utils.compute_metrics_nozero(spatial_data)
    #utils.plot_umap(envi_model.spatial_data.obsm['envi_latent'], envi_model.sc_data.obsm['envi_latent'], save="envi_latent")

    utils.save_results_all(spatial_data, args.out_file_path, args.experiment_name, metric_list=["Pearson", "Spearman", "Kendall_tau", "norm_RMSE", "RMSE"])

    spatial_data.write_loom("{}/ENVI_ST_{}.loom".format(args.out_file_path, args.experiment_name), write_obsm_varm=True)
    seq_data.write_loom("{}/ENVI_SC_{}.loom".format(args.out_file_path, args.experiment_name), write_obsm_varm=True)



