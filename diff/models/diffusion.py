import torch
import numpy as np
from functools import partial
from typing import Tuple, List
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data

from .scheduler import beta_schedule
from .tools import sliced_logsumexp, \
    log_sub_exp, to_torch_const, extract, index_to_log_onehot

   

class Diffusion(nn.Module):
    def __init__(self, timesteps:int):
        super(Diffusion, self).__init__()
        self.timesteps = timesteps
        self.scaling = 1.0  # Scaling factor for the noise.

    @staticmethod
    def get_betas(timesteps:int, **kwargs) -> torch.Tensor:
        """
        Returns the beta schedule tensor based on the given type.
        
        Args:
            timesteps (int): The number of time steps.
            type (str): The type of beta schedule.
        
        Returns:
            torch.Tensor: The beta schedule tensor.
        """
        if kwargs["type"] == 'advance':
            betas = beta_schedule(timesteps, **kwargs)
        elif kwargs["type"] == 'segment':
            betas = beta_schedule(timesteps, **kwargs)
        else:
            raise ValueError(f"Invalid beta schedule type: {kwargs['type']}")
        return betas
    
    def qrt(self, 
            r: torch.Tensor, 
            t: int, 
            beta_config:dict={
                            'type': 'advance',
                            'scale_start': 0.9999,
                            'scale_end': 0.0001,
                            'width': 3}
        ) -> torch.Tensor:
        """
        Adds noise to the ATOMIC POSITIONS, the noise is modeled as a Gaussian distribution.
        
        q(r^{t}_i | r^{t-1}_i) := \mathcal{N}(r^{t}_i | \sqrt{1-\beta^{t} r^{t-1}_i}, \beta^{t}\mathbb{I})
        - $\mathbb{I}^{3x3}$ is the identity matrix.
        - $\beta^t \in [0,1]$ (noise scaling schedule).

        Args:
            r (torch.Tensor): The atomic positions tensor.
            t (int): The current time step.
            beta_config (dict): The beta configuration.
        
        Returns:
            torch.Tensor: The noisy atomic positions tensor.
        """
        betas = self.get_betas(self.timesteps, **beta_config)
        alphas = 1. - betas
        alphas_hat = torch.cumprod(alphas, dim=0)
        alpha_t = alphas_hat[t].reshape(1, -1)
        noise = torch.randn_like(r)
        noisy_r = torch.sqrt(alpha_t) * r + torch.sqrt(1 - alpha_t) * noise
        return noisy_r


class Multinomial_diffussion(nn.Module):
    """Class for Multinomial diffussion (Categorical features)
    https://github.com/alexaoh/tabular-diffusion-for-counterfactuals/blob/master/Multinomial_diffusion.py

    Args:
        categorical_feature_names (List[str]): List of categorical feature names.
        timesteps (int): Number of timesteps.
        categorical_levels list: list of number of levels of each categorical feature. Should be in the same order as categorical_feature_names.
    """
    
    def __init__(self, 
                categorical_feature_names: List[str],
                timesteps: int,
                categorical_levels: List[int],
                device: str = 'cpu'):
        super(Multinomial_diffussion, self).__init__()
        self.categorical_feature_names = categorical_feature_names
        self.categorical_levels = categorical_levels
        self.device = device
        assert len(categorical_levels) == len(categorical_feature_names), \
                            f"Categorical levels {categorical_levels} and features names {categorical_feature_names} must be two lists of same length."
        
        self.num_categorical_features = len(self.categorical_feature_names)
        self.num_classes_extended = torch.from_numpy(
            np.concatenate([np.repeat(i, level) for i, level in enumerate(self.categorical_levels)])
        ).to(device)


        slices_for_classes = [[] for _ in range(len(self.categorical_feature_names))]
        slices_for_classes[0] = np.arange(self.categorical_levels[0])
        offsets = np.cumsum(self.categorical_levels)
        for i in range(1, self.num_categorical_features):
            slices_for_classes[i] = np.arange(offsets[i-1], offsets[i])
        self.slices_for_classes = slices_for_classes
        print(self.slices_for_classes)
        offsets = np.append([0], offsets) # Add 0 at the beginning
        self.offsets = torch.from_numpy(offsets).to(device)
        
        self.T = timesteps
        betas = self.get_betas(
            self.T, type='segment',
            **{
                'time_segment': [600, 400],
                'segment_diff': [
                    {'scale_start': 0.9999, 'scale_end': 0.001, 'width': 3},
                    {'scale_start': 0.001, 'scale_end': 0.0001, 'width': 2}
                ]
            }
                               )  # Ensure this method exists and works as expected
        
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        one_minus_alpha_bar = 1 - alpha_bar
        sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)
        sqrt_recip_alpha = torch.sqrt(1 / alphas)
        sqrt_recip_one_minus_alpha_bar = torch.sqrt(1. / one_minus_alpha_bar)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])
        
        # Logarithmic versions
        log_alphas = torch.log(alphas)
        log_one_minus_alphas = torch.log(1 - torch.exp(log_alphas) + 1e-40)
        log_alpha_bar = torch.log(alpha_bar)
        log_one_minus_alpha_bar = torch.log(1 - torch.exp(log_alpha_bar) + 1e-40)
        
        beta_tilde = betas * (1.0 - alpha_bar_prev) / (1. - alpha_bar)
        
        to_torch = partial(torch.tensor, device=device, dtype=torch.float32)
            
        # params Gaussian diffusion
        self.register_buffer("betas", to_torch(betas).to(self.device))
        self.register_buffer("alphas", to_torch(alphas).to(self.device))
        self.register_buffer("alpha_bar", to_torch(alpha_bar).to(self.device))
        self.register_buffer("sqrt_alpha_bar", to_torch(sqrt_alpha_bar).to(self.device))
        self.register_buffer("one_minus_alpha_bar", to_torch(one_minus_alpha_bar).to(self.device))
        self.register_buffer("sqrt_one_minus_alpha_bar", to_torch(sqrt_one_minus_alpha_bar).to(self.device))
        self.register_buffer("sqrt_recip_alpha", to_torch(sqrt_recip_alpha).to(self.device))
        self.register_buffer("sqrt_recip_one_minus_alpha_bar", to_torch(sqrt_recip_one_minus_alpha_bar).to(self.device))

        # Parameters for Multinomial diffusion.
        self.register_buffer("log_alphas", to_torch(log_alphas).to(self.device))
        self.register_buffer("log_one_minus_alphas", to_torch(log_one_minus_alphas).to(self.device))
        self.register_buffer("log_alpha_bar", to_torch(log_alpha_bar).to(self.device))
        self.register_buffer("log_one_minus_alpha_bar", to_torch(log_one_minus_alpha_bar).to(self.device))
        self.register_buffer("beta_tilde", to_torch(beta_tilde).to(self.device))
    
    @staticmethod
    def get_betas(timesteps:int, **kwargs) -> torch.Tensor:
        """
        Returns the beta schedule tensor based on the given type.
        
        Args:
            timesteps (int): The number of time steps.
            type (str): The type of beta schedule.
        
        Returns:
            torch.Tensor: The beta schedule tensor.
        """
        if kwargs["type"] == 'advance':
            betas = beta_schedule(timesteps, **kwargs)
        elif kwargs["type"] == 'segment':
            betas = beta_schedule(timesteps, **kwargs)
        else:
            raise ValueError(f"Invalid beta schedule type: {kwargs['type']}")
        return betas
    
    def noise_t(self, log_x_t_1: torch.Tensor, t: int) -> torch.Tensor:
        """
        Adds noise to the logit space.
        
        Args:
            log_x_t_1 (torch.Tensor): The logits at time step t-1.
            t (int): The current time step.
        
        Returns:
            torch.Tensor: The noisy logits at time step t.
        """
        log_alpha_t = extract(self.log_alphas, t, log_x_t_1.shape)
        log_one_minus_alpha_t = extract(self.log_one_minus_alphas, t, log_x_t_1.shape)
        
        log_prob = torch.logaddexp(
            log_alpha_t + log_x_t_1, log_one_minus_alpha_t - torch.log(self.num_classe_extended)
            ) # TODO: search equation in the paper
        return log_prob

    def noise_data_point(self, log_x_0: torch.Tensor, t: int) -> torch.Tensor:
        """
        Return the log of the new probability that is used in the categorical distribution
        to sample x_t.
        
        q(x_t | x_0) from eq 12 in Hoogeboom et al.
        """
        log_alpha_bar_t = extract(self.log_alpha_bar, t, log_x_0.shape)
        log_one_minus_alpha_bar_t = extract(self.log_one_minus_alpha_bar, t, log_x_0.shape)
        log_prob = torch.logaddexp(
            log_alpha_bar_t + log_x_0, log_one_minus_alpha_bar_t - torch.log(self.num_classes_extended)
            )
        return log_prob
    
    def theta_post(self, log_x_t: torch.Tensor, log_x_0: torch.Tensor, t: int) -> torch.Tensor:
        """
        This is the probability parameter in the posterior categorical distribution, called theta_post by Hoogeboom et. al.
        """
        # TabDDPM implementation: Here they removed all negative zeros from t-1, i.e. set them equal to zero.
        log_probs_x_t_1 = self.noise_data_point(log_x_0, t-1) # [\bar \alpha_{t-1} x_0 + (1-\bar \alpha_{t-1})/K]

        # t = 0 and t != 0
        num_axes = (1,) * (len(log_x_0.size()) -1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_0) # broadcast t to the same shape as log_x_0

        # if t == 0, then log_probs_x_t_1 = log_x_0
        log_probs_x_t_1 = torch.where(t_broadcast == 0, log_x_0, log_probs_x_t_1.to(torch.float32))

        log_tilde_theta = log_probs_x_t_1 + self.noise_t(log_x_t, t) # \tilde{\theta} probability vector.

        normalizing_constant = sliced_logsumexp(log_tilde_theta, self.offsets)

        normalized_log_tilde_theta = log_tilde_theta - normalizing_constant
        return normalized_log_tilde_theta
    
    def reverse_pred(self, model_pred: torch.Tensor, log_x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Returns the probability parameter of the categorical distribution of the backward process,
          based on the predicted x_0 from the neural net."""
        #hat_x_0 = self.model(log_x_t,t) # Predict x_0 from the model. 
        # We keep the one-hot-encoding of log_x_t and feed it like that into the model. 

        assert model_pred.size(0) == log_x_t.size(0), "The batch size of the model prediction and log_x_t must be the same."
        assert model_pred.size(1) == sum(self.categorical_levels), "The number of classes in model prediction and log_x_t must be the same."

        log_hat_x_0 = torch.empty_like(model_pred)

        for ix in self.slices_for_classes:
            log_hat_x_0[:, ix] = F.log_softmax(model_pred[:, ix], dim=-1) # log_softmax for each categorical variable.
        
        log_tilde_theta_hat = self.theta_post(log_hat_x_0, log_x_t, t) # fin the probability parameter of the categorical distribution of the backward process.
        return log_tilde_theta_hat
    
    def forward_sample(self, log_x_0: torch.Tensor, t: int) -> torch.Tensor:
        """Sample a new data point from q(x_t|x_0). Returns log x_t (noised input x_0 at times t, in log space), 
        
        following the closed form Equation 12 in Hoogeboom et al. q(x_t|x_0).
        """
        log_prob = self.noise_data_point(log_x_0, t)
        log_x_t = self.log_sample_categorical(log_prob)
        return log_x_t

    def log_sample_categorical(self, log_prob: torch.Tensor) -> torch.Tensor:
        """Sample from a categorical distribution in log-space."""
        full_sample = []
        print(self.slices_for_classes)
        for i in range(len(self.categorical_feature_names)):
 
            log_probs_one_cat_var = log_prob[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(log_probs_one_cat_var)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-40) + 1e-40)

            sample = (log_probs_one_cat_var + gumbel_noise).argmax(dim=1) #1 or -1
            full_sample.append(sample)
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_one_hot(full_sample, self.categorical_levels)
    
    def sample(self, model: nn.Module, n: int, y_dist: torch.distributions.Distribution = None) -> torch.Tensor:
        """Sample 'n' new data points from 'model'.
        
        This follows Algorithm 4 in our Master'r thesis. # TODO: reference to the algorithm in the thesis.
        """
        model.eval()

        if model.is_class_cond:
            if y_dist is None:
                raise Exception("The model is class conditional, but no class distribution was provided.")
        
        y = None
        if model.is_class_cond:
            y = torch.multinomial(
                y_dist,
                num_samples=n,
                replacement=True
            ).to(self.device)
        
        with torch.no_grad():
            uniform_sample = torch.zeros((n, len(self.num_classe_extended)), device=self.device)
            log_x = self.log_sample_categorical(uniform_sample).to(self.device)

            for i in reversed(range(self.T)):               # for t = T-1, T-2, ..., 1, 0
                if i % 25 == 0:
                    print(f"Sampling time step {i}")
                
                t = (torch.ones(n, device=self.device) * i).to(torch.int64)

                log_x_hat = model(log_x, t, y)
                log_tilde_theta_hat = self.reverse_pred(log_x_hat, log_x, t)
                log_x = self.log_sample_categorical(log_tilde_theta_hat)

        x = torch.exp(log_x)
        model.train()                       # indicate that we are done sampling
        if model.is_class_cond:
            y = y.reshape(-1, 1)
        return torch.cat([x, y], dim=1)

    def categorical_kl(self, log_prob_a: torch.Tensor, log_prob_b: torch.Tensor) -> torch.Tensor:
        """Calculate the KL divergence between two categorical distributions in log-space."""
        return torch.sum(torch.exp(log_prob_a) * (log_prob_a - log_prob_b), dim=1)
    
    def loss(self, log_x_0: torch.Tensor, log_x_t: torch.Tensor, log_hat_x_0: torch.Tensor, t: int) -> torch.Tensor:
        """Function to return the loss. This loss represents each term L_{t-1} in the ELBO of diffusion models.
        
        KL( q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t) ) = KL( Cat(\pi(x_t,x_0)) || Cat(\pi(x_t, \hatx_0)) ).

        We also need to compute the term log p(x_0|x_1) if t = 1. This is done via a mask below.   """

        log_true_theta = self.theta_post(log_x_t, log_x_0, t)
        log_hat_theta = self.reverse_pred(log_hat_x_0, log_x_t, t)
        lt = self.categorical_kl(log_true_theta, log_hat_theta)

        mask = (t == torch.zeros_like(t)).to(torch.float32)

        decoder_loss = -(torch.exp(log_x_0) * log_hat_x_0).sum(dim=1)

        loss = mask * decoder_loss + (1. - mask) * lt
        return torch.mean(loss)