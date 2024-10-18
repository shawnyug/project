import torch

from scvi import REGISTRY_KEYS
from scvi.module import Classifier
from scvi.train import TrainingPlan
from _module import DCVAE


class CTLTrainingPlan(TrainingPlan):
    """constrastive training plan."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        """Training step."""
        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts
        # batch contains both data loader outputs
        total_loss_list = []
        kl_list = []
        rec_loss_list = []
        contra_loss_list = []

        # first input scRNA data to network
        for (key, tensor_list) in batch.items():
            loss_output_objs = []
            n_obs = 0
            zs = []
            qz = []
            qmu = []
            topK_mask = []
            corr_ind = []
            for i, tensor in enumerate(tensor_list):
                n_obs += tensor[i].n_obs
                self.loss_kwargs.update({"kl_weight": self.kl_weight, "mode": i})
                inference_kwargs = {"mode": i}
                generative_kwargs = {"mode": i}
                ## MODIFY here the input tensor shoule be a list of tensors.
                #print("iteration {}".format(i))
                #print(batch[i][REGISTRY_KEYS.X_KEY])
                #print("the input tuple for model is:", batch)
                inference_outputs, _, loss_output = self.forward(
                    tensor, # here the batch is tuple with 2 dict containing scRNA and ST data respectively
                    loss_kwargs=self.loss_kwargs,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                )
                zs.append(inference_outputs["z"])
                #q_attent.append(inference_outputs["q"]) #here q is encoder last layer used to sampling mean and var
                qz.append(inference_outputs["qz"])
                qmu.append(inference_outputs["qmu"])
                topK_mask.append(inference_outputs["topK_mask"])
                corr_ind.append(inference_outputs["corr_ind"])
                loss_output_objs.append(loss_output)

            contra_loss = DCVAE.contrast_loss(self.module, zs[0], zs[1], key, corr_ind)

            loss = sum([scl.loss for scl in loss_output_objs])

            loss /= n_obs
            loss = loss + 5*contra_loss
            rec_loss = sum([scl.reconstruction_loss_sum for scl in loss_output_objs])
            kl = sum([scl.kl_local_sum for scl in loss_output_objs])

            print(f"{key} forward, total_loss:{loss:.3f} | contra_loss:{contra_loss*5:.3f} | mmd_loss:{mmd_loss:.3f} | kl_loss:{(kl/n_obs):.3f} | rec_loss:{(rec_loss/n_obs):.3f}")
            #print(f"{key} forward, total_loss:{loss:.3f} | contra_loss:{contra_loss:.3f} | cross_kl:{torch.sum(cross_modal_kl)/n_obs:.3f} | kl_loss:{(kl / n_obs):.3f} | rec_loss:{(rec_loss / n_obs):.3f}")


            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()

            total_loss_list.append(loss)
            kl_list.append(kl)
            rec_loss_list.append(rec_loss)
            contra_loss_list.append(contra_loss)
        print("*"*50)
        return_dict = {
            "loss": sum(total_loss_list),
            "reconstruction_loss_sum": sum(rec_loss_list),
            "kl_local_sum": sum(kl_list),
            "contrastive_loss": sum(contra_loss_list),
            "kl_global": 0.0
            #"n_obs": n_obs,
        }

        return return_dict

    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Validation step."""
        self.loss_kwargs.update({"kl_weight": self.kl_weight, "mode": dataloader_idx})
        inference_kwargs = {"mode": dataloader_idx}
        generative_kwargs = {"mode": dataloader_idx}
        _, _, loss_output = self.forward(
            batch,
            loss_kwargs=self.loss_kwargs,
            inference_kwargs=inference_kwargs,
            generative_kwargs=generative_kwargs,
        )
        reconstruction_loss = loss_output.reconstruction_loss_sum
        self.validation_step_outputs.append(
            {
                "reconstruction_loss_sum": reconstruction_loss,
                "kl_local_sum": loss_output.kl_local_sum,
                "kl_global": loss_output.kl_global,
                "n_obs": loss_output.n_obs_minibatch,
            }
        )

    def on_validation_epoch_end(self):
        """Aggregate validation step information."""
        super().on_validation_epoch_end()
        outputs = self.validation_step_outputs
        n_obs, elbo, rec_loss, kl_local = 0, 0, 0, 0
        for val_metrics in outputs:
            elbo += val_metrics["reconstruction_loss_sum"] + val_metrics["kl_local_sum"]
            rec_loss += val_metrics["reconstruction_loss_sum"]
            kl_local += val_metrics["kl_local_sum"]
            n_obs += val_metrics["n_obs"]
        # kl global same for each minibatch
        self.log("elbo_validation", elbo / n_obs)
        self.log("reconstruction_loss_validation", rec_loss / n_obs)
        self.log("kl_local_validation", kl_local / n_obs)
        self.log("kl_global_validation", 0.0)
        self.validation_step_outputs.clear()  # free memory
