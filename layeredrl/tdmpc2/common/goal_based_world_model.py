from copy import deepcopy

import numpy as np
import torch

from . import layers, math, init

from .world_model import WorldModel


class GoalBasedWorldModel(WorldModel):
    """
    Goal-based TD-MPC2 implicit world model architecture.
    The goal can be anything the policy, the reward function, and the Q-function
    can be conditioned on. In particular skill vectors can be used instead of
    goals as well.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # encode goal like state
        self._goal_encoder = layers.enc(cfg, input_shape=cfg.goal_shape)
        # add goal to input of networks
        self._reward = layers.mlp(
            cfg.latent_dim + cfg.action_dim + cfg.task_dim + cfg.latent_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
        )
        self._pi = layers.mlp(
            cfg.latent_dim + cfg.task_dim + cfg.latent_dim,
            2 * [cfg.mlp_dim],
            2 * cfg.action_dim,
        )
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    cfg.latent_dim + cfg.action_dim + cfg.task_dim + cfg.latent_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)

    def encode_goal(self, g, task):
        """
        Encodes a goal into its latent representation.
        This implementation assumes a single state-based goal.
        """
        return self._goal_encoder["state"](g)

    def reward(self, z, zg, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a, zg], dim=-1)
        return self._reward(z)

    def pi(self, z, zg, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, zg], dim=-1)

        # Gaussian policy prior
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask:  # Mask out unused action dimensions
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, zg, a, task, return_type="min", target=False):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
                - `min`: return the minimum of two randomly subsampled Q-values.
                - `avg`: return the average of two randomly subsampled Q-values.
                - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {"min", "avg", "all"}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a, zg], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)

        if return_type == "all":
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == "min" else (Q1 + Q2) / 2
