from matplotlib import cm, colors
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager  # to solve: Font family ['sans-serif'] not found. Falling back to DejaVu Sans.
import json
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy import stats
import pandas as pd
import numpy as np

import scanpy as sc
from anndata import AnnData

import os

centimeter = 1 / 2.54  # centimeter in inches

# https://www.geeksforgeeks.org/react-js-blueprint-colors-qualitative-color-schemes/
react_cols_10 = ['#147EB3', '#29A634', '#D1980B', '#D33D17', '#9D3F9D', '#00A396', '#DB2C6F', '#8EB125', '#946638',
                 '#7961DB']

# http://tsitsul.in/blog/coloropt/
norm_7 = ['#4053d3', '#ddb310', '#b51d14', '#00beff', '#fb49b0', '#00b25d', '#cacaca']
norm_12 = ['#ebac23', '#b80058', '#008cf9', '#006e00', '#00bbad', '#d163e6', '#b24502',
           '#ff9287', '#5954d6', '#00c6f8', '#878500', '#00a76c', '#bdbdbd']


def config_rc(dpi=300, font_size=6, lw=1.):
    # matplotlib.rcParams.keys()
    rc = {
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'figure.dpi': dpi, 'axes.linewidth': lw,
    }  # 'figure.figsize':(11.7/1.5,8.27/1.5)

    sns.set(style='ticks', rc=rc)
    sns.set_context("paper")

    mpl.rcParams.update(rc)

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    # mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['axes.unicode_minus'] = False  # negative minus sign


def get_path(key, json_path='./_data.json'):
    with open(json_path, 'r') as f:
        x = json.loads(f.read())
    return x[key]


def _draw_palette(cols):
    sns.color_palette(cols)


from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

def cv_data_gen(genelist, cv_mode="CV", kfold=10):
    """ Generates pair of training/test gene indexes cross validation datasets

    Args:
        genelist (list): list of all shared genes by adata_sc and adata_sp
        mode (str): Optional. support 'loo' and '10fold'. Default is 'loo'.

    Yields:
        tuple: list of train_genes, list of test_genes
    """

    #genes_array = np.array(adata_sp.uns["training_genes"])
    genes_array = np.array(genelist)

    if cv_mode == "loo":
        cv = LeaveOneOut()
    elif cv_mode == "CV":
        cv = KFold(n_splits=kfold)

    for train_idx, test_idx in cv.split(genes_array):
        train_genes = list(genes_array[train_idx])
        test_genes = list(genes_array[test_idx])
        yield train_genes, test_genes


def get_coef(X, y):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(X.reshape(-1,1), y)
    return reg.coef_[0], reg.intercept_

def plot_calibration_curve(mat1, mat2, idx, workdir, save):
    config_rc(dpi=200, font_size=10)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8 * centimeter, 8 * centimeter), sharex=True, sharey=True)

    for gene in idx:  # rand_test_gene_idx  rand_train_gene_idx
        y_test = mat1[:, gene]
        y_prob = mat2[:, gene]
        # y_prob = ad_ge[:,gene].X
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5)
        coef_, intercept_ = get_coef(prob_pred, prob_true)
        disp = CalibrationDisplay(prob_true, prob_pred, y_prob)
        disp.plot(ax=ax)

    plt.savefig(os.path.join(workdir, f"{save}.pdf"), bbox_inches='tight', format='pdf', dpi=200)

def plot_umap(latent_seq, latent_spatial, save):
    latent_representation = np.concatenate([latent_seq, latent_spatial])
    latent_adata = AnnData(latent_representation)

    # labels which cells were from the sequencing dataset and which were from the spatial dataset
    latent_labels = (["seq"] * latent_seq.shape[0]) + (
            ["spatial"] * latent_spatial.shape[0]
    )
    latent_adata.obs["labels"] = latent_labels

    # compute umap
    sc.pp.neighbors(latent_adata, use_rep="X", metric="correlation")  #
    sc.tl.umap(latent_adata)

    config_rc(dpi=200, font_size=10)
    sc.pl.umap(latent_adata, color="labels", show=False, save=f"{save}.pdf")


def compute_metrics(adata):
    auc = []
    coef_genes = []
    inter_genes= []

    for v1, v2 in zip(adata.layers["X_binary"].T, adata.obsm["zero_prob"].T):  # iterate every col of G and G_predicted
        auc.append(roc_auc_score(v1, v2))
        prob_true, prob_pred = calibration_curve(v1, v2, n_bins=5)
        coef_, intercept_ = get_coef(prob_pred, prob_true)
        coef_genes.append(coef_)
        inter_genes.append(intercept_)

    pearson = []
    spearman = []
    kendall= []
    RMSE = []

    for v1, v2 in zip(adata.X.T, adata.obsm["imputed"].T):
        personR = stats.pearsonr(v1.reshape(-1), v2.reshape(-1))
        spearmanR = stats.spearmanr(v1.reshape(-1), v2.reshape(-1))
        kendallTau = stats.kendalltau(v1.reshape(-1), v2.reshape(-1))
        rmse = mean_squared_error(v1, v2, squared=False)

        pearson.append(personR[0])
        spearman.append(spearmanR[0])
        kendall.append(kendallTau[0])
        RMSE.append(rmse)

    norm_raw = stats.zscore(adata.X, axis=0)
    norm_imputed = stats.zscore(adata.obsm["imputed"], axis=0)

    norm_rmse = []
    for v1, v2 in zip(norm_raw.T, norm_imputed.T):
        rmse = mean_squared_error(v1, v2, squared=False)
        norm_rmse.append(rmse)

    adata.var["AUC"] = auc
    adata.var["calib_slope"] = coef_genes
    adata.var["calib_intercept"] = inter_genes
    adata.var["Pearson"] = pearson
    adata.var["Spearman"] = spearman
    adata.var["Kendall_tau"] = kendall
    adata.var["norm_RMSE"] = norm_rmse
    adata.var["RMSE"] = RMSE


def compute_metrics_nozero(adata):

    pearson = []
    spearman = []
    kendall= []
    RMSE = []

    for v1, v2 in zip(adata.X.T, adata.obsm["imputed"].T):
        personR = stats.pearsonr(v1.reshape(-1), v2.reshape(-1))
        spearmanR = stats.spearmanr(v1.reshape(-1), v2.reshape(-1))
        kendallTau = stats.kendalltau(v1.reshape(-1), v2.reshape(-1))
        rmse = mean_squared_error(v1, v2, squared=False)

        pearson.append(personR[0])
        spearman.append(spearmanR[0])
        kendall.append(kendallTau[0])
        RMSE.append(rmse)

    norm_raw = stats.zscore(adata.X, axis=0)
    norm_imputed = stats.zscore(adata.obsm["imputed"], axis=0)

    norm_rmse = []
    for v1, v2 in zip(norm_raw.T, norm_imputed.T):
        rmse = mean_squared_error(v1, v2, squared=False)
        norm_rmse.append(rmse)

    adata.var["Pearson"] = pearson
    adata.var["Spearman"] = spearman
    adata.var["Kendall_tau"] = kendall
    adata.var["norm_RMSE"] = norm_rmse
    adata.var["RMSE"] = RMSE


def save_results(adata, train_genes, test_genes, result_dir, sample_id, metric_list = ["AUC", "calib_slope", "calib_intercept", "Pearson", "Spearman", "Kendall_tau", "norm_RMSE", "RMSE"]):
    # check whether the metrics result.csv file exits
    result_file = "{}/result.csv".format(result_dir)
    if os.path.exists(result_file):
        result_df = pd.read_csv(result_file, index_col=0)
        result_df[f"{sample_id}_train"] = adata.var.loc[train_genes, metric_list].mean().values
        result_df[f"{sample_id}_test"] = adata.var.loc[test_genes, metric_list].mean().values
    else:
        result_df = pd.DataFrame({f"{sample_id}_train": adata.var.loc[train_genes, metric_list].mean(), f"{sample_id}_test": adata.var.loc[test_genes, metric_list].mean()})

    result_df.to_csv("{}/result.csv".format(result_dir))

def save_results_all(adata, result_dir, sample_id, metric_list = ["AUC", "calib_slope", "calib_intercept", "Pearson", "Spearman", "Kendall_tau", "norm_RMSE", "RMSE"]):
    # check whether the metrics result.csv file exits
    result_file = "{}/result.csv".format(result_dir)
    if os.path.exists(result_file):
        result_df = pd.read_csv(result_file, index_col=0)
        result_df[f"{sample_id}_CV"] = adata.var.loc[:, metric_list].mean().values
    else:
        result_df = pd.DataFrame({f"{sample_id}_CV": adata.var.loc[:, metric_list].mean()})

    result_df.to_csv("{}/result.csv".format(result_dir))
