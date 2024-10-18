"""Main module."""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
from torch.nn import ModuleList

from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data

from nn import Encoder, MultiDecoder, one_hot, MultiEncoderCrossAttention

torch.backends.cudnn.benchmark = True


class DCVAE(BaseModuleClass):

    def __init__(
        self,
        dim_input_list: List[int],
        total_genes: int,
        indices_mappings: List[Union[np.ndarray, slice]],
        gene_likelihoods: List[str],
        model_library_bools: List[bool],
        library_log_means: List[Optional[np.ndarray]],
        library_log_vars: List[Optional[np.ndarray]],
        corr_matrix: Optional[np.ndarray],
        topK_binary_matrix_sc,
        topK_binary_matrix_st,
        cross_attention: bool = True,
        fix_attention_weights: bool = False,
        n_latent: int = 20,
        topK_contrastive: int = 50,
        n_layers_encoder_individual: int = 1,
        n_layers_encoder_shared: int = 1,
        dim_hidden_encoder: int = 64,
        n_layers_decoder_individual: int = 0,
        n_layers_decoder_shared: int = 0,
        dim_hidden_decoder_individual: int = 64,
        dim_hidden_decoder_shared: int = 64,
        dropout_rate_encoder: float = 0.2,
        dropout_rate_decoder: float = 0.2,
        n_batch: int = 0,
        n_labels: int = 0,
        dispersion: str = "gene-batch",
        log_variational: bool = True,
    ):
        super().__init__()

        self.n_input_list = dim_input_list
        self.total_genes = total_genes
        self.indices_mappings = indices_mappings
        self.gene_likelihoods = gene_likelihoods
        self.model_library_bools = model_library_bools
        for mode in range(len(dim_input_list)):
            if self.model_library_bools[mode]:
                self.register_buffer(
                    f"library_log_means_{mode}",
                    torch.from_numpy(library_log_means[mode]).float(),
                )
                self.register_buffer(
                    f"library_log_vars_{mode}",
                    torch.from_numpy(library_log_vars[mode]).float(),
                )

        self.n_latent = n_latent

        self.n_batch = n_batch
        self.n_labels = n_labels

        self.dispersion = dispersion
        self.log_variational = log_variational

        #self.corr_matrix = torch.from_numpy(corr_matrix).float().to("cuda")
        self.corr_matrix = corr_matrix
        smooth_corr = torch.nn.Softplus(beta=1)
        self.soft_corr = smooth_corr(self.corr_matrix)

        self.topK_binary_matrix_sc = topK_binary_matrix_sc
        self.topK_binary_matrix_st = topK_binary_matrix_st

        self.topK_binary_matrix = ((topK_binary_matrix_sc==1) | (topK_binary_matrix_st==1)).long()
        ### change to and
        #self.topK_binary_matrix = ((topK_binary_matrix_sc==1) & (topK_binary_matrix_st==1)).long()

        self.topK_contrastive = topK_contrastive
        
        ## change MultiEncoder to MultiEncoderCrossAttention
        self.z_encoder = MultiEncoderCrossAttention(
            n_heads=len(dim_input_list),
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            n_layers_shared=n_layers_encoder_shared,
            dropout_rate=dropout_rate_encoder,
            return_dist=True,
            cross_attention=cross_attention,
            fix_attention_weights=fix_attention_weights
        )

        self.l_encoders = ModuleList(
            [
                Encoder(
                    self.n_input_list[i],
                    1,
                    n_layers=1,
                    dropout_rate=dropout_rate_encoder,
                    return_dist=True,
                )
                if self.model_library_bools[i]
                else None
                for i in range(len(self.n_input_list))
            ]
        )

        self.decoder = MultiDecoder(
            self.n_latent,
            self.total_genes,
            n_hidden_conditioned=dim_hidden_decoder_individual,
            n_hidden_shared=dim_hidden_decoder_shared,
            n_layers_conditioned=n_layers_decoder_individual,
            n_layers_shared=n_layers_decoder_shared,
            n_cat_list=[self.n_batch],
            dropout_rate=dropout_rate_decoder,
        )

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(self.total_genes, n_labels))
        else:  # gene-cell
            pass

    def sample_from_posterior_z(
        self, x: List[torch.Tensor], M: List[torch.Tensor], mode: int = None, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample tensor of latent values from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, n_latent)``
        """
        if mode is None:
            if len(self.n_input_list) == 1:
                mode = 0
            else:
                raise Exception("Must provide a mode when having multiple datasets")
        outputs = self.inference(x, M, mode)
        qz_m = outputs["qz"].loc
        z = outputs["z"]
        if deterministic:
            z = qz_m
        return z
        
    @auto_move_data    
    def get_shared_q(self, x: List[torch.Tensor], M: List[torch.Tensor], mode: Optional[int] = None) -> torch.Tensor:
        """Sample tensor of latent values from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, n_latent)``
        """
        """Run the inference model."""
        x_ = x
        if self.log_variational:            
            x_ = [torch.log(1 + t) for t in x_]

        qz, z, q = self.z_encoder(x_, M, mode)
        return q
        
    def sample_from_posterior_l(
        self, x: List[torch.Tensor], M: List[torch.Tensor], mode: int = None, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample the tensor of library sizes from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, 1)``
        """
        inference_out = self.inference(x, M, mode)
        return (
            inference_out["ql"].loc
            if (deterministic and inference_out["ql"] is not None)
            else inference_out["library"]
        )

    def sample_scale(
        self,
        x: List[torch.Tensor],
        M: List[torch.Tensor],
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        decode_mode: Optional[int] = None,
    ) -> torch.Tensor:
        """Return the tensor of predicted frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        mode
            int encode mode (which input head to use in the model)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)``
        deterministic
            bool - whether to sample or not
        decode_mode
            int use to a decode mode different from encoding mode

        Returns
        -------
        type
            tensor of predicted expression
        """
        gen_out = self._run_forward(
            x,
            M,
            mode,
            batch_index,
            y=y,
            deterministic=deterministic,
            decode_mode=decode_mode,
        )
        return gen_out["px_scale"]

    # This is a potential wrapper for a vae like get_sample_rate
    def get_sample_rate(self, x, batch_index, *_, **__):
        """Get the sample rate for the model."""
        return self.sample_rate(x, 0, batch_index)

    ### modify: change JVAE forward tensor as list
    def _run_forward(
        self,
        x: List[torch.Tensor],
        M: List[torch.Tensor],
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        decode_mode: int = None,
    ) -> dict:
        """Run the forward pass of the model."""
        if decode_mode is None:
            decode_mode = mode
        inference_out = self.inference(x, M, mode)
        if deterministic:
            z = inference_out["qz"].loc
            if inference_out["ql"] is not None:
                library = inference_out["ql"].loc
            else:
                library = inference_out["library"]
        else:
            z = inference_out["z"]
            library = inference_out["library"]
        gen_out = self.generative(z, library, batch_index, y, decode_mode)
        return gen_out

    def sample_rate(
        self,
        x: List[torch.Tensor],
        M: List[torch.Tensor],
        mode: int,
        batch_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        decode_mode: int = None,
    ) -> torch.Tensor:
        """Returns the tensor of scaled frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        y
            tensor of cell-types labels with shape ``(batch_size, n_labels)``
        mode
            int encode mode (which input head to use in the model)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        deterministic
            bool - whether to sample or not
        decode_mode
            int use to a decode mode different from encoding mode

        Returns
        -------
        type
            tensor of means of the scaled frequencies
        """
        gen_out = self._run_forward(
            x,
            M,
            mode,
            batch_index,
            y=y,
            deterministic=deterministic,
            decode_mode=decode_mode,
        )
        return gen_out["px_rate"]

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        """Compute the reconstruction loss."""
        reconstruction_loss = None
        if self.gene_likelihoods[mode] == "zinb":
            reconstruction_loss = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=px_r, zi_logits=px_dropout
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "nb":
            reconstruction_loss = (
                -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
            )
        elif self.gene_likelihoods[mode] == "poisson":
            reconstruction_loss = -Poisson(px_rate).log_prob(x).sum(dim=1)
        return reconstruction_loss

    ### modify the inference input to a list
    # original is a dict by retriving 'X' from the dict, here tensors are a tuple of two dict, **need to return a dict containing a list of dataset**
    # add sample index to the dataloader
    def _get_inference_input(self, tensors):
        """Get the input for the inference model."""
        # get the corresponding correlation matrix for batches in each modality, shape: batch_size_modality1 x batch_size_modality2
        # ind_row = tensors[0][REGISTRY_KEYS.INDICES_KEY].squeeze()#.to(torch.long)
        # ind_col = tensors[1][REGISTRY_KEYS.INDICES_KEY].squeeze()#.to(torch.long)

        # M = self.corr_matrix
        # M1 = M.index_select(0, ind_row)
        # M2 = M1.index_select(1, ind_col)
        M = self._get_correlation_matrix(tensors)
        
        return {"x": [tensors[0].X.float(), tensors[1].X.float()], "M": M}

    def _get_correlation_matrix(self, tensors):
        """Get the input for the inference model."""
        # get the corresponding correlation matrix for batches in each modality, shape: batch_size_modality1 x batch_size_modality2
        ind_row = tensors[0].obs['index'].squeeze()#.to(torch.long)
        ind_col = tensors[1].obs['index'].squeeze()#.to(torch.long)
        
        # M1 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)
        # M2 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)
        self.sub_topK_binary_sc = self.topK_binary_matrix_sc.index_select(0, ind_row).index_select(1, ind_col)
        self.sub_topK_binary_st = self.topK_binary_matrix_st.index_select(0, ind_row).index_select(1, ind_col)
        self.corr_ind = [ind_row, ind_col]
        self.topK_mask = [self.sub_topK_binary_st, self.sub_topK_binary_sc]

        M1 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)
        M2 = self.corr_matrix.index_select(0, ind_row).index_select(1, ind_col)

        return ([M1,M2])

    def _get_generative_input(self, tensors, inference_outputs):
        """Get the input for the generative model."""
        z = inference_outputs["z"]
        library = inference_outputs["library"]

        ## modify add mode (not used in add label in latent z)
        mode = inference_outputs["mode"]

        ## random give a number to tuple tensor
        batch_index = tensors[mode].obs['batch']
        y = tensors[mode].obs['labels']
        return {"z": z, "library": library, "batch_index": batch_index, "y": y}

    ## modify the inference function input as a list of tensors
    @auto_move_data
    def inference(self, x: List[torch.Tensor], M: List[torch.Tensor], mode: Optional[int] = None) -> dict:
        """Run the inference model."""
        x_ = x
        if self.log_variational:            
            x_ = [torch.log(1 + t) for t in x_]

        qz, qmu, z, q = self.z_encoder(x_, M, mode)
        ql, library = None, None
        if self.model_library_bools[mode]:
            ql, library = self.l_encoders[mode](x_[mode])
        else:
            library = torch.log(torch.sum(x[mode], dim=1)).view(-1, 1)
        # add mode to inference output
        return {"qz": qz, "qmu": qmu, "z": z, "ql": ql, "library": library, "mode":mode, "q": q, "topK_mask": self.topK_mask[mode], "corr_ind": self.corr_ind[mode]}

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mode: Optional[int] = None,
    ) -> dict:
        """Run the generative model."""
        ## here the batch_index and y is not used in gimVI model
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            z, mode, library, self.dispersion, batch_index, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index.reshape(batch_index.shape[0],-1), self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r.view(1, self.px_r.size(0))

        px_r = torch.exp(px_r)
        px_scale = px_scale / torch.sum(
            px_scale[:, self.indices_mappings[mode]], dim=1
        ).view(-1, 1)
        px_rate = px_scale * torch.exp(library)

        return {
            "px_scale": px_scale,
            "px_r": px_r,
            "px_rate": px_rate,
            "px_dropout": px_dropout,
        }

    def loss(
        self,
        tensors, ## modify: input is a list of tensors now
        inference_outputs,
        generative_outputs,
        mode: Optional[int] = None,
        kl_weight=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the reconstruction loss and the Kullback divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape (batch_size, n_labels)
        mode
            indicates which head/tail to use in the joint network


        Returns
        -------
        the reconstruction loss and the Kullback divergences
        """
        if mode is None:
            if len(self.n_input_list) == 1:
                mode = 0
            else:
                raise Exception("Must provide a mode")
        ## get the corrresponding tensor from tuple
        tensors = tensors[mode]
        x = tensors.X
        batch_index = tensors.obs['batch']

        qz = inference_outputs["qz"]
        ql = inference_outputs["ql"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        # mask loss to observed genes
        mapping_indices = self.indices_mappings[mode]

        ##DEBUG
        #print("mapping_indices len is: ", len(mapping_indices))
        #print("mapping_indices", mapping_indices)
        #print("mode is:", mode)
        #print("px_rate shape {}, and the input shape is {}".format(px_rate.shape, x.shape))
        #print("px_r shape {}, and the input shape is {}".format(px_r.shape, x.shape))
        reconstruction_loss = self.reconstruction_loss(
            x,
            px_rate[:, mapping_indices],
            px_r[:, mapping_indices],
            px_dropout[:, mapping_indices],
            mode,
        )

        # KL Divergence
        mean = torch.zeros_like(qz.loc)
        scale = torch.ones_like(qz.scale)
        kl_divergence_z = kl(qz, Normal(mean, scale)).sum(dim=1)

        if self.model_library_bools[mode]:
            library_log_means = getattr(self, f"library_log_means_{mode}")
            library_log_vars = getattr(self, f"library_log_vars_{mode}")

            local_library_log_means = F.linear(
                one_hot(batch_index.reshape(batch_index.shape[0],-1), self.n_batch), library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index.reshape(batch_index.shape[0],-1), self.n_batch), library_log_vars
            )
            kl_divergence_l = kl(
                ql,
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        kl_local = kl_divergence_l + kl_divergence_z

        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) * x.size(0)

        return LossOutput(
            loss=loss, reconstruction_loss=reconstruction_loss, kl_local=kl_local
        )

    def contrast_loss(self, emb_sc, emb_st, label, corr_ind):
        M_sim = F.cosine_similarity(emb_sc[:, None, :], emb_st[None, :, :], dim=-1)
        ind_row, ind_col = corr_ind
        sub_corr = self.soft_corr.index_select(0, ind_row).index_select(1, ind_col)
        sub_topK_binary = self.topK_binary_matrix.index_select(0, ind_row).index_select(1, ind_col)
        #sub_topK_binary_st, sub_topK_binary_sc = topK_mask
        #loss = 0
        if label == "st":
            sub_topK_binary_sc = self.topK_binary_matrix_sc.index_select(0, ind_row).index_select(1, ind_col)
            # get the top rank cell-spot pairs
            # for a given spot, get the topK cell index
            pos_sc = torch.sum(sub_topK_binary_sc, dim=0)  # calculate the num of positive pairs for each spot
            loss = -torch.sum(torch.mul(M_sim, sub_topK_binary_sc) / sub_corr, dim=0) + torch.mul(torch.logsumexp(M_sim * torch.nn.functional.softmax(~sub_topK_binary_sc / sub_corr, dim=0), dim=0),pos_sc)
            # loss1 = -torch.sum(M_sim * sub_topK_binary_sc / sub_corr1, dim=0) + torch.mul(torch.logsumexp(M_sim * ~sub_topK_binary_sc / sub_corr1, dim=0), pos_sc)
            #loss = -torch.sum(torch.mul(M_sim, sub_topK_binary), dim=0) + torch.mul(torch.logsumexp(torch.mul(M_sim, ~sub_topK_binary), dim=0), pos_sc)
            #loss1 = loss1.mean()  # * M_sim.shape[1]
        
        elif label == "sc":
            sub_topK_binary_st = self.topK_binary_matrix_st.index_select(0, ind_row).index_select(1, ind_col)
            # for a given cell, get the topK spot index
            pos_st = torch.sum(sub_topK_binary_st, dim=1)  # calculate the num of positive pairs for each cell
            loss = -torch.sum(torch.mul(M_sim, sub_topK_binary_st) / sub_corr, dim=1) + torch.mul(torch.logsumexp(M_sim * torch.nn.functional.softmax(~sub_topK_binary_st / sub_corr, dim=1), dim=1),pos_st)
            # loss2 = -torch.sum(M_sim * sub_topK_binary_st / sub_corr2, dim=1) + torch.mul(torch.logsumexp(M_sim * ~sub_topK_binary_st / sub_corr2, dim=1), pos_st)
            #loss = -torch.sum(torch.mul(M_sim, sub_topK_binary), dim=1) + torch.mul(torch.logsumexp(torch.mul(M_sim, ~sub_topK_binary), dim=1), pos_st)
            #loss2 = loss2.mean()  # * M_sim.shape[0]

        return loss.mean()


    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def compute_mmd(self, source, target, kernel_mul=2, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                                  fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

