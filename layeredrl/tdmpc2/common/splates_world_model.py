from copy import deepcopy

import numpy as np
import torch

from . import layers, math, init

from .world_model import WorldModel


class SPlaTESWorldModel(WorldModel):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.control_interval = cfg.control_interval
        self._reward = layers.mlp(
            cfg.latent_dim
            + cfg.action_dim
            + cfg.task_dim
            + cfg.skill_dim
            + cfg.abstract_latent_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
        )
        self._pi = layers.mlp(
            cfg.latent_dim
            + cfg.task_dim
            + cfg.time_dim
            + cfg.skill_dim
            + cfg.abstract_latent_dim,
            2 * [cfg.mlp_dim],
            2 * cfg.action_dim,
        )
        self._value_averaged = layers.mlp(
            cfg.latent_dim + cfg.task_dim + cfg.time_dim + cfg.abstract_latent_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
        )
        # self._reward_average = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    cfg.latent_dim
                    + cfg.action_dim
                    + cfg.task_dim
                    + cfg.time_dim
                    + cfg.skill_dim
                    + cfg.abstract_latent_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )
        # for converting TDMPC2 latent to abstract latent
        self._latent_converter = layers.mlp(
            cfg.latent_dim,
            2 * [cfg.mlp_dim],
            cfg.abstract_latent_dim,
        )
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)

    def reward(self, z, a, s_start, skill, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a, s_start, skill], dim=-1)
        return self._reward(z)

    def pi(self, z, k, s_start, skill, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        mu, log_std = self._pi(
            torch.cat([z, k / self.control_interval, s_start, skill], dim=-1)
        ).chunk(2, dim=-1)
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

    def Q(self, z, a, k, s_start, skill, task, return_type="min", target=False):
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

        z = torch.cat([z, a, k / self.control_interval, s_start, skill], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)

        if return_type == "all":
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == "min" else (Q1 + Q2) / 2

    def V_avg(self, z, k, s_start, task, return_type):
        """
        Predicts the state value averaged over all skills.
        `return_type` can be one of [`logits`, `avg`]:
                - `logits`: return the logits of the value distribution.
                - `avg`: return the average of the value distribution.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, k / self.control_interval, s_start], dim=-1)
        if return_type == "logits":
            return self._value_averaged(z)
        else:
            return math.two_hot_inv(self._value_averaged(z), self.cfg)

    def latent_converter(self, z, task):
        """
        Converts the TD-MPC2 latent to abstract latent.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        return self._latent_converter(z)
