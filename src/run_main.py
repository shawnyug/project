import os
import argparse
import time
import random

import numpy as np
import scanpy as sc
from scanpy import logging as logg
import scvi
from scvi.data import cortex, smfish
from _model import STFormer
import torch

import utils

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='imputeST model')
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--data_path", type=str, default="./data/imputation/DataUpload", required=False)
parser.add_argument("--data_id", type=str, default="Dataset9", required=False)
parser.add_argument('--batch_size', default=1024, type=int, help='number of batch_size')
parser.add_argument("--min_cells", type=int, default=1)
parser.add_argument("--min_genes", type=int, default=1)

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out_file_path", type=str, default="../results", required=False)

parser.add_argument("--gpu_device", type=str, default="cuda")
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=300)

parser.add_argument("--generative_dist", type=str, nargs='+', required=True)
parser.add_argument("--model_library_size", type=bool, nargs='+', required=True)
parser.add_argument("--n_encoder_layers_individual", type=int, default=1)
parser.add_argument("--n_encoder_layers_shared", type=int, default=1)
parser.add_argument("--dim_hidden_encoder", type=int, default=64)
parser.add_argument("--n_encoder_neurons", type=int, default=64)
parser.add_argument("--latent_dim", type=int, default=10)
parser.add_argument("--n_decoder_layers_individual", type=int, default=0)
parser.add_argument("--n_decoder_layers_shared", type=int, default=0)
parser.add_argument("--dim_hidden_decoder_individual", type=int, default=64)
parser.add_argument("--dim_hidden_decoder_shared", type=int, default=64)
#parser.add_argument("--topK", type=int, default=50)
parser.add_argument("--topK_contrastive", type=int, default=50)
parser.add_argument("--topK_sampler", type=int, default=50)
parser.add_argument("--corr_metric", type=str, default="Pearson")
parser.add_argument("--cross_attention", action="store_true", default=False, required=False)
parser.add_argument("--fix_attention_weights", action="store_true", default=False, required=False)

parser.add_argument("--ground_truth_index", type=str, default=None,
                    help="specify the true label index of the spots")
parser.add_argument("--barcode_index", type=str, default="index", help="specify the barcode index of the spots")
parser.add_argument("--spot_size", type=float, default=None)
parser.add_argument("--scanpy_process", action="store_true", default=False, required=False)
parser.add_argument("--use_gpu", action="store_true", default=False, required=False)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_data(data_path, data_id):
    spatial_data = smfish()
    seq_data = cortex()
    # in_file_path_list = [os.path.join(in_file_path, "{}.h5ad".format(sample_id)) for sample_id in
    #                      sample_id_list]
    # for dataset one, seqFISH hippocampus data

    #return seq_data, spatial_data_partial

def preprocess():
    pass

if __name__ == '__main__':
    # parse arg
    args = parser.parse_args()

    if args.experiment_name:
        out_file_path = os.path.join(args.out_file_path, args.experiment_name)
        if not os.path.isdir(out_file_path):
            os.mkdir(out_file_path)
        args.out_file_path = out_file_path

    sc.settings.figdir = args.out_file_path
    sc.settings.set_figure_params(vector_friendly=True)

    # load & process data
    data_root = os.path.join("./data", args.data_id)
    seq_data = sc.read_h5ad(f"{data_root}_sc.h5ad")
    spatial_data = sc.read_h5ad(f"{data_root}_st.h5ad")

    train_size = 0.8
    spatial_data.var_names = [x.lower() for x in spatial_data.var_names]
    seq_data.var_names = [x.lower() for x in seq_data.var_names]

    spatial_data.var_names_make_unique()
    seq_data.var_names_make_unique()

    #sc.pp.calculate_qc_metrics(seq_data, percent_top=None, log1p=False, inplace=True)
    #mt_genes = seq_data.var_names.str.startswith("mt-")
    #seq_data = seq_data[seq_data.obs["total_counts"] <= 33000, ~mt_genes].copy()
    #sc.pp.filter_genes(seq_data, min_cells=int(0.01 * seq_data.shape[0]))
    #seq_data.X = seq_data.X.toarray()

    #sc.pp.calculate_qc_metrics(spatial_data, percent_top=None, log1p=False, inplace=True)
    #spatial_data = spatial_data[spatial_data.obs["total_counts"] <= 600, :].copy()

    # # remove cells with no counts
    # sc.pp.filter_cells(spatial_data, min_counts=100)
    # sc.pp.filter_cells(seq_data, min_counts=100)
    #
    # sc.pp.filter_genes(spatial_data, min_cells=1)
    # sc.pp.filter_genes(seq_data, min_cells=1)

    # subset spatial data into shared genes
    gene_names = np.intersect1d(spatial_data.var_names, seq_data.var_names)

    # only use genes in both datasets
    seq_data = seq_data[:, gene_names].copy()
    spatial_data = spatial_data[:, gene_names].copy()

    seq_gene_names = seq_data.var_names
    n_genes = seq_data.n_vars
    n_train_genes = int(n_genes * train_size)

    # set the seed for random and torch
    #seed_torch(args.seed)

    # randomly select training_genes
    rand_train_gene_idx = np.random.choice(range(n_genes), n_train_genes, replace=False)
    rand_test_gene_idx = sorted(set(range(n_genes)) - set(rand_train_gene_idx))
    rand_train_genes = seq_gene_names[rand_train_gene_idx]
    rand_test_genes = seq_gene_names[rand_test_gene_idx]

    spatial_data.uns["meta"] = {"train_genes": rand_train_genes, "test_genes": rand_test_genes, "train_idx": rand_train_gene_idx, "test_idx": rand_test_gene_idx }

    # spatial_data_partial has a subset of the genes to train on
    spatial_data_partial = spatial_data[:, rand_train_genes].copy()


    # spatial_data should use the same cells as our training data
    # cells may have been removed by scanpy.pp.filter_cells()
    spatial_data = spatial_data[spatial_data_partial.obs_names]

    seq_data.obs["index"] = np.arange(seq_data.shape[0])
    spatial_data_partial.obs["index"] = np.arange(spatial_data_partial.shape[0])

    seq_data.obs["batch"] = 0
    spatial_data_partial.obs["batch"] = 1
    seq_data.obs["labels"] = 0
    spatial_data_partial.obs["labels"] = 1

    ## scale spatial coordinates for ST
    #pos_scaler = MinMaxScaler()
    #pos2d = spatial_data_partial.obsm["spatial"]
    #pos2d = np.array(spatial_data_partial.obs[['x_coord', 'y_coord']])
    #spatial_data_partial.obsm["spatial"] = pos_scaler.fit_transform(pos2d)
    # spatial_data_partial.obsm["spatial"] = np.array(spatial_data_partial.obs[['x_coord', 'y_coord']])

    use_cuda = torch.cuda.is_available()
    if use_cuda and args.use_gpu:
        device = args.gpu_device
        logg.info(f'\nPut the data and model into GPU: {device}')
    else:
        device = "cpu"
        logg.info('\nPut the data and model into CPU')

    start = time.time()

    # create our model
    model = STFormer(adata_seq=seq_data, adata_spatial=spatial_data_partial,
                     device=args.gpu_device,
                     generative_distributions=args.generative_dist,
                     model_library_size=args.model_library_size,
                     n_latent=args.latent_dim,
                     n_layers_encoder_individual=args.n_encoder_layers_individual,
                     n_layers_encoder_shared=args.n_encoder_layers_shared,
                     n_layers_decoder_individual=args.n_decoder_layers_individual,
                     n_layers_decoder_shared=args.n_decoder_layers_shared,
                     cross_attention=args.cross_attention,
                     fix_attention_weights=args.fix_attention_weights,
                     corr_metric=args.corr_metric,
                     topK_contrastive=args.topK_contrastive)

    # train for 200 epochs
    model.train(max_epochs=args.epochs,
                batch_size=args.batch_size,
                plan_kwargs = {"lr": args.lr}, 
                use_gpu=args.gpu_device)
    end = time.time()
    print('running time = {}'.format(end - start))

    sc_dropout, st_dropout = model.get_px_para(args.batch_size, "px_dropout")
    sc_imputed, st_imputed = model.get_px_para(args.batch_size, "px_rate")
    sc_theta, st_theta = model.get_px_para(args.batch_size, "px_r")
    #sc_pos, st_pos = model.get_px_para(args.batch_size, "pos")

    latent_seq, latent_spatial = model.get_latent_representation()
    latent_seq_val, latent_spatial_val = model.get_latent_val_representation()

    # whether to slice ST data when num of ST spots is smaller than cells
    n_samples = [seq_data.shape[0], spatial_data_partial.shape[0]]
    ind = np.argmin(n_samples)
    if ind==0:
        sc_dropout = sc_dropout[:n_samples[ind],:]
        sc_imputed = sc_imputed[:n_samples[ind],:]
        sc_theta = sc_theta[:n_samples[ind],:]
        #sc_pos = sc_pos[:n_samples[ind],:]
        latent_seq = latent_seq[:n_samples[ind],:]
        latent_seq_val = latent_seq_val[:n_samples[ind],:]
    elif ind==1:
        st_dropout = st_dropout[:n_samples[ind],:]
        st_imputed = st_imputed[:n_samples[ind],:]
        st_theta = st_theta[:n_samples[ind],:]
        #st_pos = st_pos[:n_samples[ind],:]
        latent_spatial = latent_spatial[:n_samples[ind],:]
        latent_spatial_val = latent_spatial_val[:n_samples[ind],:]

    # zero probability for NB in ST
    nb_zero_prob = np.power((st_theta / (st_theta + st_imputed)), st_theta)


    def binary(gene):
        # ind = np.where(gene==0.0)
        gene_copy = gene.copy()
        gene[~(gene_copy == 0.0)] = 0
        gene[(gene_copy == 0.0)] = 1
        return gene

    seq_data.obsm["imputed"] = sc_imputed
    seq_data.obsm["theta"] = sc_theta
    seq_data.obsm["dropout"] = sc_dropout
    seq_data.obsm["latent_train"] = latent_seq
    seq_data.obsm["latent_val"] = latent_seq_val
    #seq_data.obsm["infer_spatial"] = sc_pos

    spatial_data.layers["X_binary"] = np.apply_along_axis(binary, 0, spatial_data.copy().X)
    spatial_data.obsm["zero_prob"] = nb_zero_prob
    spatial_data.obsm["imputed"] = st_imputed
    spatial_data.obsm["theta"] = st_theta
    spatial_data.obsm["dropout"] = st_dropout
    spatial_data.obsm["latent_train"] = latent_spatial
    spatial_data.obsm["latent_val"] = latent_spatial_val
    #spatial_data.obsm["infer_spatial"] = st_pos

    utils.compute_metrics(spatial_data)
    utils.plot_calibration_curve(spatial_data.layers["X_binary"], spatial_data.obsm["zero_prob"], rand_train_gene_idx,
                                 workdir=args.out_file_path, save=f"{args.experiment_name}_train_calibration")
    utils.plot_calibration_curve(spatial_data.layers["X_binary"], spatial_data.obsm["zero_prob"], rand_test_gene_idx,
                                 workdir=args.out_file_path, save=f"{args.experiment_name}_test_calibration")
    utils.plot_umap(latent_seq, latent_spatial, save="train")
    utils.plot_umap(latent_seq_val, latent_spatial_val, save="validation")
    utils.save_results(spatial_data, rand_train_genes, rand_test_genes, args.out_file_path, args.experiment_name)

    spatial_data.write_loom("{}/imputeFormer_ST_{}.loom".format(args.out_file_path, args.experiment_name), write_obsm_varm=True)
    seq_data.write_loom("{}/imputeFormer_SC_{}.loom".format(args.out_file_path, args.experiment_name), write_obsm_varm=True)
    #anndata.read_loom("spatial_data", var_names='var_names', obs_names="obs_names")




