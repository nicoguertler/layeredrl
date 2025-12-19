import torch
import torch.nn.functional as F

from .tdmpc2 import TDMPC2
from .common import math
from .common.goal_based_world_model import GoalBasedWorldModel


class GoalBasedTDMPC2(TDMPC2):
    """
    Goal-based TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg, device: torch.device = torch.device("cpu")):
        super().__init__(cfg, device)
        self.model = GoalBasedWorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {
                    "params": self.model._goal_encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._Qs.parameters()},
                {
                    "params": (
                        self.model._task_emb.parameters() if self.cfg.multitask else []
                    )
                },
                {"params": self.model._term_prob.parameters()},
            ],
            lr=self.cfg.lr,
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
        )
        self.model.eval()

    @torch.no_grad()
    def act(self, obs, g, t0=False, eval_mode=False, task=None, use_policy=False):
        """
        Select an action by planning in the latent space of the world model.

        Args:
                obs (torch.Tensor): Observation from the environment.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (int): Task index (only used for multi-task experiments).
                task (bool): Whether to use the policy or the planner.

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        # NOTE: Assume there already is a batch dimension
        # obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(obs, task)
        zg = self.model.encode_goal(g, task)
        if self.cfg.mpc and not use_policy:
            a = self.plan(z, zg, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, zg, task)[int(not eval_mode)][0]
        return a

    @torch.no_grad()
    def _estimate_value(self, z, zg, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions.

        Note that z and actions are assumed to have a batch dimension."""
        G = 0
        discount = torch.ones(*z.shape[:2], 1, device=z.device)
        _zg = zg[:, None, :].expand(-1, z.shape[1], -1)
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(
                self.model.reward(z, _zg, actions[:, t], task), self.cfg
            )
            # take probability of environment termination into account
            cont_prob = 1.0 - self.model.term_prob(z, actions[:, t], task)
            z = self.model.next(z, actions[:, t], task)
            G += discount * reward
            discount *= (
                self.discount[torch.tensor(task)]
                if self.cfg.multitask
                else self.discount
            )
            discount *= cont_prob
        return G + discount * self.model.Q(
            z, _zg, self.model.pi(z, _zg, task)[1], task, return_type="avg"
        )

    @torch.no_grad()
    def plan(self, z, zg, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.

        Note: This version has a BATCH DIMENSION (for environment instances).

        Args:
                z (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        bsz = z.shape[0]
        # Sample policy trajectories
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(
                bsz,
                self.cfg.horizon,
                self.cfg.num_pi_trajs,
                self.cfg.action_dim,
                device=self.device,
            )
            _z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
            _zg = zg.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[:, t] = self.model.pi(_z, _zg, task)[1]
                _z = self.model.next(_z, pi_actions[:, t], task)
            pi_actions[:, -1] = self.model.pi(_z, _zg, task)[1]

        # Initialize state and parameters
        z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
        mean = torch.zeros(
            bsz, self.cfg.horizon, self.cfg.action_dim, device=self.device
        )
        std = self.cfg.max_std * torch.ones(
            bsz, self.cfg.horizon, self.cfg.action_dim, device=self.device
        )
        if (~t0).any() and self._prev_mean is not None:
            # shift initialization
            mean[~t0, :-1] = self._prev_mean[~t0, 1:]
        actions = torch.empty(
            bsz,
            self.cfg.horizon,
            self.cfg.num_samples,
            self.cfg.action_dim,
            device=self.device,
        )
        if self.cfg.num_pi_trajs > 0:
            actions[:, :, : self.cfg.num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            actions[:, :, self.cfg.num_pi_trajs :] = (
                mean.unsqueeze(2)
                + std.unsqueeze(2)
                * torch.randn(
                    bsz,
                    self.cfg.horizon,
                    self.cfg.num_samples - self.cfg.num_pi_trajs,
                    self.cfg.action_dim,
                    device=std.device,
                )
            ).clamp(-1, 1)
            if self.cfg.multitask:
                raise NotImplementedError(
                    "Multi-task MPPI not implemented in batched version."
                )
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, zg, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(
                value.squeeze(2), self.cfg.num_elites, dim=1
            ).indices
            # elite_value, elite_actions = value[:, elite_idxs], actions[:, :, elite_idxs]
            elite_value = value.gather(1, elite_idxs.unsqueeze(-1))
            elite_actions = actions.gather(
                2,
                elite_idxs[:, None, :, None].expand(
                    -1, self.cfg.horizon, -1, self.cfg.action_dim
                ),
            )

            # Update parameters
            max_value = elite_value.max(1, keepdim=True)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(
                1, keepdim=True
            )  # Don't need this when using multinomial...
            mean = torch.sum(score.unsqueeze(1) * elite_actions, dim=2) / (
                score.sum(1, keepdim=True) + 1e-9
            )
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(1) * (elite_actions - mean.unsqueeze(2)) ** 2, dim=2
                )
                / (score.sum(1, keepdim=True) + 1e-9)
            ).clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                raise NotImplementedError(
                    "Multi-task MPPI not implemented in batched version."
                )
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        score = score.squeeze(2)  # .cpu().numpy()
        # actions = elite_actions[:, :, np.random.choice(np.arange(score.shape[1]), p=score)]
        actions = elite_actions.gather(
            2,
            torch.multinomial(score, 1)[:, :, None, None].expand(
                -1, self.cfg.horizon, -1, self.cfg.action_dim
            ),
        )
        actions = actions.squeeze(2)
        self._prev_mean = mean
        a, std = actions[:, 0], std[:, 0]
        if not eval_mode:
            a += std * torch.randn(bsz, self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)

    def update_pi(self, zs, zg, task):
        """
        Update policy using a sequence of latent states.

        Args:
                zs (torch.Tensor): Sequence of latent states.
                zg (torch.Tensor): Goal.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, zg, task)
        qs = self.model.Q(zs, zg, pis, task, return_type="avg")
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm
        )
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_z, zg, reward, terminated, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
                next_z (torch.Tensor): Latent state at the following time step.
                zg (torch.Tensor): Embedded goal.
                reward (torch.Tensor): Reward at the current time step.
                terminated (torch.Tensor): Whether the episode has terminated.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, zg, task)[1]
        discount = (
            self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        )
        discount = discount * (1.0 - terminated.unsqueeze(-1).float())
        return reward + discount * self.model.Q(
            next_z, zg, pi, task, return_type="min", target=True
        )

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
                buffer (common.buffer.Buffer): Replay buffer.

        Returns:
                dict: Dictionary of training statistics.
        """
        obs, g, action, reward, terminated, task = buffer.sample()

        zg = self.model.encode_goal(g, task)

        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            # which goals are used has to be consistent with her implementation in tdmpc2_level.py
            td_targets = self._td_target(next_z, zg[:-1], reward, terminated, task)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(
            self.cfg.horizon + 1,
            obs.shape[1],
            self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t + 1] = z

        # Predictions
        _zs = zs[:-1]
        _zgs = zg[:-1]  # again consistency with her implementation
        qs = self.model.Q(_zs, _zgs, action, task, return_type="all")
        reward_preds = self.model.reward(_zs, _zgs, action, task)
        term_probs = self.model.term_prob(_zs, action, task)

        # Compute losses
        reward_loss, value_loss, term_loss = 0, 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += (
                math.soft_ce(reward_preds[t], reward[t], self.cfg).mean()
                * self.cfg.rho**t
            )
            for q in range(self.cfg.num_q):
                value_loss += (
                    math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean()
                    * self.cfg.rho**t
                )
            term_loss += F.binary_cross_entropy(
                term_probs[t], terminated[t].unsqueeze(-1).float()
            )
        consistency_loss *= 1 / self.cfg.horizon
        reward_loss *= 1 / self.cfg.horizon
        value_loss *= 1 / (self.cfg.horizon * self.cfg.num_q)
        term_loss *= 1 / self.cfg.horizon
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
            + self.cfg.term_coef * term_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()

        # Update policy
        pi_loss = self.update_pi(zs.detach(), zg.detach(), task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }
