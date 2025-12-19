from typing import Callable, Union, Optional

from gymnasium.spaces import Box
import torch
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from .predictor import Predictor
from ..models import Model


class PredictorFactory:
    """A factory class creating predictors taking the latent state and context dimension into account."""

    def __init__(
        self,
        partial_model: Callable[..., Model],
        partial_val_func: Optional[Callable[..., Union[Module, Callable]]],
        partial_rew_func: Callable[..., Union[Module, Callable]],
        partial_encoder: Callable[..., Union[Module, Callable]],
        partial_predictor: Callable[..., Predictor],
    ):
        """Initialize the predictor factory.

        Args:
            partial_model: A partial model (expecting spaces).
            partial_val_func: A partial value function (expecting latent state and context dim).
            partial_rew_func: A partial reward function (expecting latent state, context dim, and action dim).
            partial_encoder: A partial encoder (expecting mapped env obs shape, latent state dim, and context dim).
            partial_predictor: A partial predictor (without model, value function, and map to latent).
            latent_state_dim: The dimension of the latent state space.
            context_dim: The dimension of the context variable (encoding time invariant information).
        """
        self.model_factory = partial_model
        self.val_func_factory = partial_val_func
        self.rew_func_factory = partial_rew_func
        self.encoder_factory = partial_encoder
        self.partial_predictor_factory = partial_predictor

    def __call__(
        self,
        mapped_env_obs_shape,
        action_space,
        device: torch.device,
        writer: SummaryWriter,
    ) -> Predictor:
        """Create a new predictor.

        Args:
            mapped_env_obs_shape: The shape of the mapped environment observation.
            action_space: The action space associated to the predictor.
            device: The device to use.
        Returns:
            A new predictor."""
        encoder = self.encoder_factory(mapped_env_obs_shape, device=device)
        latent_state_dim = encoder.latent_state_dim
        context_dim = encoder.context_dim
        latent_state_space = Box(
            low=-float("inf"), high=float("inf"), shape=(latent_state_dim,)
        )
        context_space = Box(low=-float("inf"), high=float("inf"), shape=(context_dim,))
        model = self.model_factory(
            latent_state_space, context_space, action_space, device=device
        )
        if self.val_func_factory is None:
            val_func = None
        else:
            val_func = self.val_func_factory(
                latent_state_dim, context_dim, device=device
            )
        action_dim = action_space.shape[0]
        rew_func = self.rew_func_factory(
            latent_state_dim, context_dim, action_dim, device=device
        )
        return self.partial_predictor_factory(
            model=model,
            val_func=val_func,
            rew_func=rew_func,
            encoder=encoder,
            latent_state_dim=latent_state_dim,
            context_dim=context_dim,
            device=device,
            writer=writer,
        )
