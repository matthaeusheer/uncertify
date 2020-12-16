import torch
from torch import dist

from uncertify.utils.custom_types import Tensor


def loss_function(self, reconstruction: Tensor, observation: Tensor, mu: Tensor, log_var: Tensor):
    if self._loss_type == 'l2':
        rec_dist = dist.Normal(reconstruction, 1.0)
    elif self._loss_type == 'l1':
        rec_dist = dist.Laplace(reconstruction, 1.0)
    else:
        raise ValueError(f'Loss type "{self._loss_type}" not supported.')

    # p(x|z)
    log_p_x_z = rec_dist.log_prob(observation)
    log_p_x_z = torch.mean(log_p_x_z, dim=(1, 2, 3))  # Per slice negative log likelihood
    slice_rec_err = log_p_x_z

    # p(z)
    z_prior = dist.Normal(0.0, 1.0)

    # q(z|x)
    z_post = dist.Normal(mu, torch.sqrt(torch.exp(log_var)))

    # KL(q(z|x), p(z))
    slice_kl_div = dist.kl_divergence(z_post, z_prior)
    slice_kl_div = torch.mean(slice_kl_div, dim=1)  # KL divergences for each slice
    slice_kl_div = self._calculate_beta(self._train_step_counter) * slice_kl_div

    # Take the mean over all batches to get batch-wise kl divergence and log likelihood compared to slice-wise
    batch_kl_div = torch.mean(slice_kl_div)
    batch_rec_err = torch.mean(log_p_x_z)

    variational_lower_bound = (-batch_kl_div + batch_rec_err) * self._n_m_factor
    total_loss = -variational_lower_bound

    return total_loss, batch_kl_div, batch_rec_err, slice_kl_div, slice_rec_err
