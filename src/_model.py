import logging
import os
import warnings
from typing import List, Optional, Union, Tuple
from math import ceil
import random
import numpy as np
import torch
from anndata import AnnData
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import (
    Sampler,
)
from anndata.experimental.pytorch import AnnLoader

from scvi import REGISTRY_KEYS, settings
from scvi.model._utils import parse_device_args
from scvi.model.base import BaseModelClass, VAEMixin
from scvi.train import Trainer
from scvi.utils._docstrings import devices_dsp
import data_processing
from lightning.pytorch.utilities import CombinedLoader
from scipy.special import softmax


import sys

sys.path.append("./")

from _module import DCVAE
from _task import CTLTrainingPlan

logger = logging.getLogger(__name__)

# Custom sampler to get proper batches instead of joined separate indices
# maybe move to multi_files
class BatchIndexSampler(Sampler):
    def __init__(self, n_obs, batch_size, shuffle=False, drop_last=False):
        self.n_obs = n_obs
        self.batch_size = batch_size if batch_size < n_obs else n_obs
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.n_obs).tolist()
        else:
            indices = list(range(self.n_obs))

        for i in range(0, self.n_obs, self.batch_size):
            batch = indices[i : min(i + self.batch_size, self.n_obs)]

            # only happens if the last batch is smaller than batch_size
            if len(batch) < self.batch_size and self.drop_last:
                continue

            yield batch

    def __len__(self):
        if self.drop_last:
            length = self.n_obs // self.batch_size
        else:
            length = ceil(self.n_obs / self.batch_size)

        return length


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

class ContrastSampler(Sampler):
    def __init__(self, sampler, batch_size, topK_ind_matrix, topK_contrast):
        """
        input
            sampler: the sampler from another modality
            batch_size: the batch size for current modality
            topK_ind_matrix: the topK index of cells/spots in the correlation matrix respect to every single spot/cell
        """
        #super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.topK_ind_matrix = topK_ind_matrix
        self.topK_contrast = topK_contrast
        self.n_obs = 0

    def __iter__(self):
        # iterating the sampler of the other modality to get the sample index
        # sampling based on the average pearson correlation after softmax, here corr_matrix denote the big pearson correlation matrix
        for ind in self.sampler:
            topK_ind = np.arange(self.topK_contrast, dtype=int)
            generate_pos_pair = np.random.choice(topK_ind, size=len(ind))
            # sample the index with replacement
            batch_ind = self.topK_ind_matrix[ind, generate_pos_pair]
            self.n_obs = self.n_obs + len(batch_ind)

            yield batch_ind

    def __len__(self):
        return self.n_obs

class CrossModalSampler(Sampler):
    def __init__(self, sampler, batch_size, corr_matrix, axis):
        """
        input
            sampler: the sampler from another modality
            batch_size: the batch size for current modality
            corr_matrix: the topK correlation matrix between current modality and the other modality
        """
        #super().__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.sampler = sampler
        self.batch_size = batch_size
        self.corr_matrix = corr_matrix
        self.axis = axis
        self.n_obs = 0

    def __iter__(self):
        # iterating the sampler of the other modality to get the sample index
        # sampling based on the average pearson correlation after softmax, here corr_matrix denote the big pearson correlation matrix
        for ind in self.sampler:
            sub_corr = np.take(self.corr_matrix, ind, axis=self.axis)
            # average and softmax
            #avg_axis = np.argmax(sub_corr.shape) #average along the larger dim
            sub_corr_mean = np.mean(sub_corr, axis=self.axis)
            sub_corr_prob = softmax(sub_corr_mean)
            all_batch_ind = np.arange(len(sub_corr_mean))
            # sample the index with replacement
            batch_ind = np.random.choice(all_batch_ind, len(ind), p=sub_corr_prob).tolist()
            self.n_obs = self.n_obs + len(batch_ind)

            yield batch_ind

    def __len__(self):
        return self.n_obs

def _unpack_tensors(tensors):
    x = tensors[REGISTRY_KEYS.X_KEY].squeeze_(0)
    batch_index = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze_(0)
    y = tensors[REGISTRY_KEYS.LABELS_KEY].squeeze_(0)
    return x, batch_index, y

def _init_library_size(
    data, n_batch: dict
) -> Tuple[np.ndarray, np.ndarray]:

    # data = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
    batch_indices = data.obs["batch"]

    library_log_means = np.zeros(n_batch)
    library_log_vars = np.ones(n_batch)

    for i_batch in np.unique(batch_indices):
        idx_batch = np.squeeze(batch_indices == i_batch)
        batch_data = data[
            idx_batch.to_numpy().nonzero()[0]
        ]  # h5ad requires integer indexing arrays.
        sum_counts = batch_data.X.sum(axis=1)
        # Operations on numpy masked array gives invalid values masked
        # masked_array(data=[-- -- 0.0 0.69314718056],
        #              mask=[True  True False False],
        masked_log_sum = np.ma.log(sum_counts)
        if np.ma.is_masked(masked_log_sum):
            warnings.warn(
                "This dataset has some empty cells, this might fail inference."
                "Data should be filtered with `scanpy.pp.filter_cells()`",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        # Return input as an array with masked data replaced by a fill value.
        log_counts = masked_log_sum.filled(0)
        library_log_means[i_batch] = np.mean(log_counts).astype(np.float32)
        library_log_vars[i_batch] = np.var(log_counts).astype(np.float32)

    return library_log_means.reshape(1, -1), library_log_vars.reshape(1, -1)


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class STFormer(VAEMixin, BaseModelClass):

    def __init__(
        self,
        adata_seq: AnnData,
        adata_spatial: AnnData,
        generative_distributions: Optional[List[str]] = None,
        model_library_size: Optional[List[bool]] = None,
        n_latent: int = 20,
        cross_attention: bool = True,
        fix_attention_weights: bool = False,
        corr_metric: str = "Pearson",
        topK_contrastive: int = 50,
        device: str = "cuda",
        **model_kwargs,
    ):
        super().__init__()
        if adata_seq is adata_spatial:
            raise ValueError(
                "`adata_seq` and `adata_spatial` cannot point to the same object. "
                "If you would really like to do this, make a copy of the object and pass it in as `adata_spatial`."
            )

        seed_torch()

        model_library_size = model_library_size or [True, True]
        generative_distributions = generative_distributions or ["zinb", "nb"]
        self.adatas = [adata_seq, adata_spatial]

        self.registries_ = []

        seq_var_names = adata_seq.var_names
        spatial_var_names = adata_spatial.var_names

        if not set(spatial_var_names) <= set(seq_var_names):
            raise ValueError("spatial genes needs to be subset of seq genes")

        spatial_gene_loc = [
            np.argwhere(seq_var_names == g)[0] for g in spatial_var_names
        ]
        spatial_gene_loc = np.concatenate(spatial_gene_loc)
        gene_mappings = [slice(None), spatial_gene_loc]
        # sum_stats = [adm.summary_stats for adm in self.adata_managers.values()]
        n_inputs = [len(seq_var_names), len(spatial_var_names)]
        total_genes = n_inputs[0]

        n_batches = 2
        #
        library_log_means = []
        library_log_vars = []
        for adata in self.adatas:
            adata_library_log_means, adata_library_log_vars = _init_library_size(
                adata, n_batches
            )
            library_log_means.append(adata_library_log_means)
            library_log_vars.append(adata_library_log_vars)

        ############ calculate the correlation matrix between scRNA-seq and ST ###########
        self.corr_matrix_raw = data_processing.calculate_corr(adata_seq, adata_spatial, spatial_var_names, corr_metric)
        #self.corr_matrix = data_processing.calculate_kendall_tau(adata_seq, adata_spatial, spatial_var_names)


        n_samples = [adata_seq.shape[0], adata_spatial.shape[0]]
        ind = np.argmin(n_samples)

        scale_topK_contrastive = topK_contrastive * np.array(n_samples)/n_samples[ind]
        self.topK_contra_sc, self.topK_contra_st = scale_topK_contrastive.astype(int)

        _, self.topK_ind_sc_contra = data_processing.get_sample_weights(corr_matrix=self.corr_matrix_raw, topK=int(scale_topK_contrastive[0]), axis=0)
        _, self.topK_ind_st_contra = data_processing.get_sample_weights(corr_matrix=self.corr_matrix_raw, topK=int(scale_topK_contrastive[1]), axis=1)

        # initialize the binary mask among topK cells for every given spot and topK spots for every given cells
        topK_binary_matrix = torch.zeros(adata_seq.shape[0], adata_spatial.shape[0])
        topK_binary_matrix_sc = topK_binary_matrix.scatter(0, torch.tensor(self.topK_ind_sc_contra), 1)
        topK_binary_matrix_st = topK_binary_matrix.scatter(1, torch.tensor(self.topK_ind_st_contra), 1)

        self.corr_matrix = torch.from_numpy(self.corr_matrix_raw ).float().to(device)
        self.topK_binary_matrix_sc = torch.gt(topK_binary_matrix_sc, 0).to(device)
        self.topK_binary_matrix_st = torch.gt(topK_binary_matrix_st,0).to(device)

        self.module = DCVAE(
            n_inputs,
            total_genes,
            gene_mappings,
            generative_distributions,
            model_library_size,
            library_log_means,
            library_log_vars,
            self.corr_matrix,
            self.topK_binary_matrix_sc,
            self.topK_binary_matrix_st,
            cross_attention=cross_attention,
            fix_attention_weights=fix_attention_weights,
            n_batch=n_batches,
            n_latent=n_latent,
            topK_contrastive=topK_contrastive,
            **model_kwargs,
        )

        self._model_summary_string = (
            "GimVI Model with the following params: \nn_latent: {}, n_inputs: {}, n_genes: {}, "
            + "n_batch: {}, generative distributions: {}"
        ).format(n_latent, n_inputs, total_genes, n_batches, generative_distributions)
        self.init_params_ = self._get_init_params(locals())

    #@devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 200,
        use_gpu: Optional[Union[str, int, bool]] = None,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        train_size: float = 1,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        batch_size: int = 1024,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):

        accelerator, devices, device = parse_device_args(
            use_gpu=use_gpu,
            accelerator=accelerator,
            devices=devices,
            return_device="torch",
        )

        self.trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **kwargs,
        )
        self.train_indices_, self.test_indices_, self.validation_indices_ = [], [], []
        train_dls, test_dls, val_dls = [], [], []

        ######preparing the dataloader for torch model
        adata_seq, adata_spatial = self.adatas

        use_cuda = torch.cuda.is_available()

        # TODO: for random sample cells, spots in cross attention and spots in contrastive learning are all conditional on randomly selected cells
        rand_sampler_sc = BatchIndexSampler(adata_seq.shape[0], batch_size=batch_size, shuffle=True, drop_last=True)
        cross_sampler_st = CrossModalSampler(rand_sampler_sc, batch_size, self.corr_matrix_raw, 0)

        contra_sampler_st = ContrastSampler(rand_sampler_sc, batch_size, self.topK_ind_st_contra, self.topK_contra_st)
        contra_cross_sampler_sc = CrossModalSampler(contra_sampler_st, batch_size, self.corr_matrix_raw, 1)

        # TODO: for random sample spots, cells in cross attention and cells in contrastive learning are all conditional on randomly selected spots
        rand_sampler_st = BatchIndexSampler(adata_spatial.shape[0], batch_size=batch_size, shuffle=True, drop_last=True)
        cross_sampler_sc = CrossModalSampler(rand_sampler_st, batch_size, self.corr_matrix_raw, 1)

        contra_sampler_sc = ContrastSampler(rand_sampler_st, batch_size, self.topK_ind_sc_contra.T, self.topK_contra_sc)
        contra_cross_sampler_st = CrossModalSampler(contra_sampler_sc, batch_size, self.corr_matrix_raw, 0)

        # construct dataloader for randomly selected cells
        dl_rand_sc = AnnLoader(adata_seq, sampler=rand_sampler_sc, batch_size=None, use_cuda="mps")
        dl_cross_st = AnnLoader(adata_spatial, sampler=cross_sampler_st, batch_size=None, use_cuda="mps")

        dl_contra_st = AnnLoader(adata_spatial, sampler=contra_sampler_st, batch_size=None, use_cuda="mps")
        dl_contra_cross_sc = AnnLoader(adata_seq, sampler=contra_cross_sampler_sc, batch_size=None, use_cuda="mps")

        # construct dataloader for randomly selected spots
        dl_rand_st = AnnLoader(adata_spatial, sampler=rand_sampler_st, batch_size=None, use_cuda="mps")
        dl_cross_sc = AnnLoader(adata_seq, sampler=cross_sampler_sc, batch_size=None,use_cuda="mps")

        dl_contra_sc = AnnLoader(adata_seq, sampler=contra_sampler_sc, batch_size=None,use_cuda="mps")
        dl_contra_cross_st = AnnLoader(adata_spatial, sampler=contra_cross_sampler_st, batch_size=None, use_cuda="mps")

        rand_sc_dl = [[dl_rand_sc, dl_cross_st], [dl_contra_cross_sc, dl_contra_st]]
        rand_st_dl = [[dl_contra_sc, dl_contra_cross_st], [dl_cross_sc, dl_rand_st]]

        train_dl = {"sc": rand_sc_dl, "st": rand_st_dl}
        train_dl = CombinedLoader(train_dl, mode="max_size_cycle")


        ## change adversarial classifier to False
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        self._training_plan = CTLTrainingPlan(
            self.module,
            **plan_kwargs,
        )

        # two dataloaders from two different datasets within-the-same-loop
        # the shorter one will keep iterating the dataset until reach the max batches of the bigger one
        # https://discuss.pytorch.org/t/two-dataloaders-from-two-different-datasets-within-the-same-loop/87766/5
        if train_size == 1.0:
            # circumvent the empty data loader problem if all dataset used for training
            self.trainer.fit(self._training_plan, train_dl)
        else:
            # accepts list of val dataloaders
            self.trainer.fit(self._training_plan, train_dl, val_dls)
        try:
            self.history_ = self.trainer.logger.history
        except AttributeError:
            self.history_ = None
        self.module.eval()

        self.to_device(device)
        self.is_trained_ = True

    def _make_scvi_dls(self, adatas: List[AnnData] = None, sample_weights: List[np.ndarray]=None, batch_size=128):
        if adatas is None:
            adatas = self.adatas
        post_list = [self._make_data_loader(ad, sampler=WeightedRandomSampler(sample_weights[i], len(sample_weights[i]))) for i,ad in enumerate(adatas)]
        for i, dl in enumerate(post_list):
            dl.mode = i

        return post_list

    @torch.inference_mode()
    def get_px_para(self, batch_size, symbol):
        # symbol is one of the values in the following:
        # px_scale: normalized gene expression frequency
        # px_rate: px_scale * exp(library)
        # px_r: dispersion parameter
        # px_dropout: dropout rate

        self.module.eval()

        # scdls = model._make_scvi_dls(model.adatas, model.sample_weights_list, batch_size=128)
        adata_seq, adata_spatial = self.adatas

        ############### load data from corresponding index for every batch in the other modality
        rand_sampler_st = BatchIndexSampler(adata_spatial.shape[0], batch_size=batch_size, shuffle=False, drop_last=False)
        rand_sampler_sc = BatchIndexSampler(adata_seq.shape[0], batch_size=batch_size, shuffle=False, drop_last=False)

        cross_sampler_sc = CrossModalSampler(rand_sampler_st, batch_size, self.corr_matrix_raw, 1)
        cross_sampler_st = CrossModalSampler(rand_sampler_sc, batch_size, self.corr_matrix_raw, 0)

        dataloader_st = AnnLoader(adata_spatial, sampler=rand_sampler_st, batch_size=None, use_cuda=True)
        dataloader_sc = AnnLoader(adata_seq, sampler=rand_sampler_sc, batch_size=None, use_cuda=True)

        dataloader_cross_st = AnnLoader(adata_spatial, sampler=cross_sampler_st, batch_size=None,use_cuda=True)
        dataloader_cross_sc = AnnLoader(adata_seq, sampler=cross_sampler_sc, batch_size=None,use_cuda=True)

        sc_dl = data_processing.TrainDL([dataloader_sc, dataloader_cross_st])
        st_dl = data_processing.TrainDL([dataloader_cross_sc, dataloader_st])

        val_dl = {"sc": sc_dl, "st": st_dl}
        ### use training dataloader
        # train_dl = TrainDL(scdls)

        retrive_values = []
        for mode, key in enumerate(["sc", "st"]):
            retrive_value = []
            dl = val_dl[key]
            for i,  batch in enumerate(dl): #
                scdl1, scdl2 = batch
                #corr_mat = model.corr_matrix
                # corr_mat = torch.from_numpy(model.corr_matrix).float().to("cuda")
                # **************** feed non-negative weights ********************#
                ind_row = scdl1.obs["index"]  # .squeeze().int().to("cuda")#.to(torch.long)
                ind_col = scdl2.obs["index"]  # .squeeze().int().to("cuda")#.to(torch.long)

                M1 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)
                M2 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)

                # M = model.module._get_correlation_matrix([scdl1, scdl2])
                dls = [scdl1, scdl2]

                retrive_value.append(
                    self.module._run_forward(
                        [scdl1.X.float(), scdl2.X.float()],
                        [M1, M2],
                        mode,
                        dls[mode].obs["batch"],  # [mode],
                        dls[mode].obs["labels"],  # [mode],
                        deterministic=True,
                        decode_mode=None,
                    )[symbol]
                )

            retrive_value = torch.cat(retrive_value).cpu().detach().numpy()
            retrive_values.append(retrive_value)

        return (retrive_values)

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adatas: List[AnnData] = None,
        deterministic: bool = True,
        batch_size: int = 1024,
    ) -> List[np.ndarray]:
        """Return the latent space embedding for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        """
        if adatas is None:
            adatas = self.adatas
        #scdls = self._make_scvi_dls(adatas, self.sample_weights_list, batch_size=batch_size)

        rand_sampler_st = BatchIndexSampler(adatas[1].shape[0], batch_size=batch_size, shuffle=False, drop_last=False)
        rand_sampler_sc = BatchIndexSampler(adatas[0].shape[0], batch_size=batch_size, shuffle=False, drop_last=False)

        cross_sampler_sc = CrossModalSampler(rand_sampler_st, batch_size, self.corr_matrix_raw, 1)
        cross_sampler_st = CrossModalSampler(rand_sampler_sc, batch_size, self.corr_matrix_raw, 0)

        dataloader_st = AnnLoader(adatas[1], sampler=rand_sampler_st, batch_size=None, use_cuda=True)
        dataloader_sc = AnnLoader(adatas[0], sampler=rand_sampler_sc, batch_size=None, use_cuda=True)

        dataloader_cross_st = AnnLoader(adatas[1], sampler=cross_sampler_st, batch_size=None,use_cuda=True)
        dataloader_cross_sc = AnnLoader(adatas[0], sampler=cross_sampler_sc, batch_size=None,use_cuda=True)

        sc_dl = data_processing.TrainDL([dataloader_sc, dataloader_cross_st])
        st_dl = data_processing.TrainDL([dataloader_cross_sc, dataloader_st])

        val_dl = {"sc": sc_dl, "st": st_dl}

        self.module.eval()
        latents = []

        for mode, key in enumerate(["sc", "st"]):
            latent = []
            dl = val_dl[key]
            for i,  batch in enumerate(dl): #
                scdl1, scdl2 = batch

                ind_row = scdl1.obs["index"]#.squeeze().int().to("cuda")#.to(torch.long)
                ind_col = scdl2.obs["index"]#.squeeze().int().to("cuda")#.to(torch.long)

                M1 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)
                M2 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)

                # M1 = self.module.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)
                # M2 = self.module.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)
                #M = self.module._get_correlation_matrix([scdl1, scdl2])
                latent.append(self.module.sample_from_posterior_z([scdl1.X.float(), scdl2.X.float()], [M1,M2], mode, deterministic=deterministic))
            latent = torch.cat(latent).cpu().detach().numpy()
            latents.append(latent)

        return latents

    @torch.inference_mode()
    def get_latent_val_representation(
            self,
            adatas: List[AnnData] = None,
            deterministic: bool = True,
            batch_size: int = 1024,
    ) -> List[np.ndarray]:
        """Return the latent space embedding for each dataset.

        Parameters
        ----------
        adatas
            List of adata seq and adata spatial.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        """
        if adatas is None:
            adatas = self.adatas
        # scdls = self._make_scvi_dls(adatas, self.sample_weights_list, batch_size=batch_size)

        ############### load data from corresponding index for every batch in the other modality
        rand_sampler_st = BatchIndexSampler(adatas[1].shape[0], batch_size=batch_size, shuffle=False, drop_last=False)
        rand_sampler_sc = BatchIndexSampler(adatas[0].shape[0], batch_size=batch_size, shuffle=False, drop_last=False)

        cross_sampler_sc = CrossModalSampler(rand_sampler_st, batch_size, self.corr_matrix_raw, 1)
        cross_sampler_st = CrossModalSampler(rand_sampler_sc, batch_size, self.corr_matrix_raw, 0)

        dataloader_st = AnnLoader(adatas[1], sampler=rand_sampler_st, batch_size=None, use_cuda=True)
        dataloader_sc = AnnLoader(adatas[0], sampler=rand_sampler_sc, batch_size=None, use_cuda=True)

        dataloader_cross_st = AnnLoader(adatas[1], sampler=cross_sampler_st, batch_size=None,use_cuda=True)
        dataloader_cross_sc = AnnLoader(adatas[0], sampler=cross_sampler_sc, batch_size=None,use_cuda=True)

        sc_dl = data_processing.TrainDL([dataloader_sc, dataloader_cross_st])
        st_dl = data_processing.TrainDL([dataloader_cross_sc, dataloader_st])

        val_dl = {"sc": sc_dl, "st": st_dl}


        self.module.eval()
        latents = []

        for mode, key in enumerate(["sc", "st"]):
            latent = []
            dl = val_dl[key]
            for i, batch in enumerate(dl):  #
                scdl1, scdl2 = batch

                ind_row = scdl1.obs["index"]  # .squeeze().int().to("cuda")#.to(torch.long)
                ind_col = scdl2.obs["index"]  # .squeeze().int().to("cuda")#.to(torch.long)

                M1 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)
                M2 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)

                latent.append(self.module.sample_from_posterior_z([scdl1.X.float(), scdl2.X.float()], [M1, M2], mode,
                                                                  deterministic=deterministic))
            latent = torch.cat(latent).cpu().detach().numpy()
            latents.append(latent)

        return latents


    def save(
        self,
        dir_path: str,
        prefix: Optional[str] = None,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        prefix
            Prefix to prepend to saved file names.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )

        file_name_prefix = prefix or ""

        seq_adata = self.adatas[0]
        spatial_adata = self.adatas[1]
        if save_anndata:
            seq_save_path = os.path.join(dir_path, f"{file_name_prefix}adata_seq.h5ad")
            seq_adata.write(seq_save_path)

            spatial_save_path = os.path.join(
                dir_path, f"{file_name_prefix}adata_spatial.h5ad"
            )
            spatial_adata.write(spatial_save_path)

        # save the model state dict and the trainer state dict only
        model_state_dict = self.module.state_dict()

        seq_var_names = seq_adata.var_names.astype(str).to_numpy()
        spatial_var_names = spatial_adata.var_names.astype(str).to_numpy()

        # get all the user attributes
        user_attributes = self._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}

        model_save_path = os.path.join(dir_path, f"{file_name_prefix}model.pt")

        torch.save(
            {
                "model_state_dict": model_state_dict,
                "seq_var_names": seq_var_names,
                "spatial_var_names": spatial_var_names,
                "attr_dict": user_attributes,
            },
            model_save_path,
        )

    @classmethod
    #@setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        index: Optional[str] = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_layer)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, index),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

# class TrainDL(DataLoader):
#     """Train data loader."""
#
#     def __init__(self, data_loader_list, **kwargs):
#         self.data_loader_list = data_loader_list
#         self.largest_train_dl_idx = np.argmax(
#             [len(dl.indices) for dl in data_loader_list]
#         )
#         self.largest_dl = self.data_loader_list[self.largest_train_dl_idx]
#         super().__init__(self.largest_dl, **kwargs)
#
#     def __len__(self):
#         return len(self.largest_dl)
#
#     def __iter__(self):
#         train_dls = [
#             dl if i == self.largest_train_dl_idx else cycle(dl)
#             for i, dl in enumerate(self.data_loader_list)
#         ]
#         return zip(*train_dls)
