import torch
import numpy as np
from functools import partial
from typing import Tuple, List
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data

from .scheduler import beta_schedule

def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def extract(a: torch.Tensor, t: int, x_shape: Tuple[int]) -> torch.Tensor:
    """Changes the dimensions of the input a depending on t and x_t.

    Makes them compatible such that pointwise multiplication of tensors can be done.
    Each column in the same row gets the same value after this "extract" function, 
    such that each data point column can be multiplied by the same number when noising and denoising. 
    """
    t = t.to(a.device)
    out = a[t]
    while len(out.shape) < len(x_shape):
        out = out[..., None] # add dimensions
    return out.expand(x_shape)
    

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
    """
    
    def __init__(self, 
                 categorical_feature_names:Tuple[str], 
                 timesteps:int,
                 categorical_levels:List[int] = None,
                 device:str = 'cpu'
                 ):
        super(Multinomial_diffussion, self).__init__()
        self.categorical_feature_names = categorical_feature_names
        self.categorical_levels = categorical_levels
        self.device = device
        assert len(categorical_levels) == len(categorical_feature_names), \
                            f"Categorical levels {categorical_levels} and features names {categorical_feature_names} must be two lists of same length."
                            
        self.num_categorical_features = len(categorical_feature_names)
        self.num_classe_extended = torch.from_numpy(
            np.concatenate([np.repeat(self.categorical_levels[i], self.categorical_levels[i]) for i in range(self.categorical_levels)])
        ).to(device)
        
        slices_for_classes = [[] for _ in range(self.num_categorical_features)]
        slices_for_classes[0] = np.arange(self.categorical_levels[0])
        offsets = np.cumsum(self.categorical_levels)
        for i in range(1, self.num_categorical_features):
            slices_for_classes[i] = np.arange(offsets[i-1], offsets[i])
        self.slices_for_classes = slices_for_classes
        offsets = np.append([0], offsets) # add 0 at the beginning
        self.offsets = torch.from_numpy(offsets).to(device)
        
        self.T = timesteps
        betas = self.get_betas(self.T, **{
            'type': 'segment',
            "scale_start": 0.9999,
            "scale_end": 0.0001,
        })
        
        alphas = 1 - betas                                              # alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)                             # alpha_bar = \prod_{i=1}^{t} (1 - \beta_i)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)                               
        one_minus_alpha_bar = 1 - alpha_bar                        
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)                 
        sqrt_recip_alpha = torch.sqrt(1 / alphas)                            
        sqrt_recip_one_minus_alpha_bar = torch.sqrt(1. / (1. - alpha_bar))  
        alpha_bar_prev = np.append(1.0, alpha_bar[:-1])                      
        
        # logarithmic versions (log space)
        log_alphas = torch.log(alphas)
        log_one_minus_alphas = torch.log(1 - torch.exp(log_alphas) + 1e-40)  # add 1e-40 to avoid log(0)
        log_alpha_bar = torch.log(alpha_bar)
        log_one_minus_alpha_bar = torch.log(1 - toch.exp(log_alpha_bar) + 1e-40)
        
        beta_tilde = betas * (1.0 - alpha_bar_prev) / (1. - alpha_bar)  # Eq 7 in DDPM paper
        
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
            log_alpha_bar_t + log_x_0, log_one_minus_alpha_bar_t - torch.log(self.num_classe_extended)
            )
        return log_prob