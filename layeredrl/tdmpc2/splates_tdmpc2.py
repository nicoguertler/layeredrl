import numpy as np
import torch
import torch.nn.functional as F

from .tdmpc2 import TDMPC2
from .common.splates_world_model import SPlaTESWorldModel
from .common import math


class SPlaTESTDMPC2(TDMPC2):

    def __init__(self, cfg, device: torch.device = torch.device("cpu")):
        super().__init__(cfg, device)
        self.model = SPlaTESWorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam(
            [
                {
                    "params": self.model._encoder.parameters(),
                    "lr": self.cfg.lr * self.cfg.enc_lr_scale,
                },
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._Qs.parameters()},
                {"params": self.model._value_averaged.parameters()},
                {
                    "params": (
                        self.model._task_emb.parameters() if self.cfg.multitask else []
                    )
                },
                {"params": self.model._latent_converter.parameters()},
                {"params": self.model._term_prob.parameters()},
            ],
            lr=self.cfg.lr,
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
        )
        self.model.eval()

    @torch.no_grad()
    def act(
        self,
        obs,
        k,
        s_start,
        skill,
        t0=False,
        eval_mode=False,
        task=None,
        use_policy=False,
    ):
        """
        Select an action by planning in the latent space of the world model.

        Args:
                obs (torch.Tensor): Observation from the environment.
                k (torch.Tensor): Time step in the skill execution.
                s_start (torch.Tensor): Starting state of the skill.
                skill (torch.Tensor): Skill vector identifying the skill that is
                    currently being executed.
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
        if self.cfg.mpc and not use_policy:
            a = self.plan(z, k, s_start, skill, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, k, s_start, skill, task)[int(not eval_mode)][0]
        return a

    @torch.no_grad()
    def _estimate_value(self, z, actions, k, s_start, skill, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions.

        Note that z and actions are assumed to have a batch dimension."""
        s_start = s_start.unsqueeze(1).expand(-1, z.shape[1], -1).clone()
        skill = skill.unsqueeze(1).expand(-1, z.shape[1], -1).clone()
        G = torch.zeros(*z.shape[:2], 1, device=z.device)
        discount = torch.ones(*z.shape[:2], 1, device=z.device)
        gamma = (
            self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
        )
        # need one more step to update k to compute the value of the last state
        for t in range(self.cfg.horizon + 1):
            current_k = k + t
            skill_not_term = current_k.squeeze(-1) < self.cfg.control_interval
            if t < self.cfg.horizon:
                reward = math.two_hot_inv(
                    self.model.reward(
                        z,
                        actions[:, t],
                        s_start,
                        skill,
                        task,
                    ),
                    self.cfg,
                )
                if skill_not_term.any():
                    cont_prob = 1.0 - self.model.term_prob(
                        z[skill_not_term], actions[skill_not_term, t], task
                    )
                    z[skill_not_term] = self.model.next(
                        z[skill_not_term], actions[skill_not_term, t], task
                    )
                    G[skill_not_term] += (
                        discount[skill_not_term] * reward[skill_not_term]
                    )
                    discount[skill_not_term] *= gamma
                    # take probability of environment termination into account
                    discount[skill_not_term] *= cont_prob
        q_bootstrap = self.model.Q(
            z,
            self.model.pi(
                z,
                current_k[:, None, :].expand(-1, z.shape[1], -1),
                s_start,
                skill,
                task,
            )[1],
            current_k[:, None, :].expand(-1, z.shape[1], -1),
            s_start,
            skill,
            task,
            return_type="avg",
        )
        if self.cfg.get("inter_skill_bootstrap", True):
            new_s_start = self.model.latent_converter(z, task)
            v_bootstrap = self.model.V_avg(
                z,
                torch.zeros(*z.shape[:-1], 1, device=self.device),
                new_s_start,
                task,
                return_type="avg",
            )
        else:
            v_bootstrap = torch.zeros_like(q_bootstrap)
        # When skill terminated, use value averaged over all skills (as next skill
        # is not known). This should induce conservative skills that are more chainable
        # than greedy skills.
        w = (skill_not_term[:, None, None]).float()
        G += discount * ((1 - w) * v_bootstrap + w * q_bootstrap)
        return G

    @torch.no_grad()
    def plan(self, z, k, s_start, skill, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.

        Note: This version has a BATCH DIMENSION (for environment instances).

        Args:
                z (torch.Tensor): Latent state from which to plan.
                k (torch.Tensor): Time step in the skill execution.
                s_start (torch.Tensor): Starting state of the skill.
                skill (torch.Tensor): Skill vector identifying the skill that is
                    currently being executed.
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
            _s_start = s_start.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
            _skill = skill.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon):
                current_k = torch.fmod(k + t, self.cfg.control_interval)
                pi_actions[:, t] = self.model.pi(
                    _z,
                    current_k[:, None, :].expand(-1, self.cfg.num_pi_trajs, -1),
                    _s_start,
                    _skill,
                    task,
                )[1]
                if t < self.cfg.horizon:
                    _z = self.model.next(_z, pi_actions[:, t], task)

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
            value = self._estimate_value(
                z, actions, k, s_start, skill, task
            ).nan_to_num_(0)
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
        score = score.squeeze(2)
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

    def update_pi(self, zs, ks, s_starts, skills, task):
        """
        Update policy using a sequence of latent states.

        Args:
                zs (torch.Tensor): Sequence of latent states.
                ks (torch.Tensor): Sequence of time steps in the skill execution.
                s_starts (torch.Tensor): Sequence of starting states of the skill.
                skills (torch.Tensor): Sequence of skill vectors.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, ks, s_starts, skills, task)
        qs = self.model.Q(zs, pis, ks, s_starts, skills, task, return_type="avg")
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
    def _td_target(
        self, next_z, next_k, next_s_start, next_skill, reward, terminated, task
    ):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
                next_z (torch.Tensor): Latent state at the following time step.
                next_k (torch.Tensor): Time step in the skill execution at the following time step.
                next_s_start (torch.Tensor): Starting state of the skill at the following time step.
                next_skill (torch.Tensor): Skill vector at the following time step.
                reward (torch.Tensor): Reward at the current time step.
                terminated (torch.Tensor): Whether the environment terminated at the current time step.
                task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_z, next_k, next_s_start, next_skill, task)[1]
        gamma = (
            self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        )
        # take environment termination into account
        discount = gamma * (1.0 - terminated.unsqueeze(-1).float())

        Q_bootstrap = self.model.Q(
            next_z,
            pi,
            next_k,
            next_s_start,
            next_skill,
            task,
            return_type="min",
            target=True,
        )
        if self.cfg.get("inter_skill_bootstrap", True):
            V_bootstrap = self.model.V_avg(
                next_z, next_k, next_s_start, task, return_type="avg"
            )
        else:
            V_bootstrap = torch.zeros_like(Q_bootstrap)
        skill_terminated = (next_k == 0).float()
        # When skill terminated, use value averaged over all skills (as next skill
        # is not known). This should induce conservative skills that are more chainable
        # than greedy skills.
        bootstrap = (
            skill_terminated * V_bootstrap + (1 - skill_terminated) * Q_bootstrap
        )
        return reward + discount * bootstrap

    @torch.no_grad()
    def _value_avg_target(self, z, k, s_start, task):
        """
        Compute the target for the value averaged over the skill space.

        Args:
                z (torch.Tensor): Latent state.
                k (torch.Tensor): Time step in the skill execution.
                s_start (torch.Tensor): Starting state of the skill.
                task (torch.Tensor): Task index (only used for multi-task experiments).
        Returns:
                torch.Tensor: Target value.
        """
        sampled_skill = (
            2.0 * torch.rand(*z.shape[:-1], self.cfg.skill_dim, device=self.device)
            - 1.0
        )
        pi = self.model.pi(z, k, s_start, sampled_skill, task)[1]
        all_qs = self.model.Q(
            z,
            pi,
            k,
            s_start,
            sampled_skill,
            task,
            return_type="all",
            target=True,
        )
        # return a random Q function as target
        return all_qs[np.random.randint(self.cfg.num_q)]

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
                buffer (common.buffer.Buffer): Replay buffer.

        Returns:
                dict: Dictionary of training statistics.
        """
        obs, action, reward, k, s_start, skill, s_abstract, terminated, task = (
            buffer.sample()
        )

        # Compute targets
        with torch.no_grad():

            initial_z = self.model.encode(obs, task)
            next_z = initial_z[1:]
            # This works because s_start and skill has already been shifted when sampling
            # from the replay buffer
            td_targets = self._td_target(
                next_z, k[1:], s_start[1:], skill[1:], reward, terminated, task
            )
            # set td target to zero for terminated states
            td_targets = td_targets * (1 - terminated.float()).unsqueeze(-1)
            value_avg_targets = self._value_avg_target(
                initial_z[:-1], k[:-1], s_start[:-1], task
            )

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.latent_dim,
            device=self.device,
        )
        ks = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            1,
            device=self.device,
        )
        s_starts = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.abstract_latent_dim,
            device=self.device,
        )
        skills = torch.empty(
            self.cfg.horizon + 1,
            self.cfg.batch_size,
            self.cfg.skill_dim,
            device=self.device,
        )
        skills = skill
        s_starts = s_start
        z = self.model.encode(obs[0], task)
        zs[0] = z
        ks[0] = k[0]
        s_starts[0] = s_start[0]
        consistency_loss = 0
        _k = k[0].clone()
        for t in range(self.cfg.horizon):
            current_k = torch.fmod(_k + t + 1, self.cfg.control_interval)
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t + 1] = z
            ks[t + 1] = current_k

        # Predictions
        _zs = zs[:-1]
        _ks = ks[:-1]
        _s_starts = s_starts[:-1]
        _skills = skills[:-1]
        qs = self.model.Q(_zs, action, _ks, _s_starts, _skills, task, return_type="all")
        reward_preds = self.model.reward(_zs, action, _s_starts, _skills, task)
        # detach to not interfere with TDMPC2 training
        v_avg = self.model.V_avg(
            _zs.detach(), _ks, _s_starts, task, return_type="logits"
        )
        s_abstract_preds = self.model.latent_converter(_zs.detach(), task)
        term_probs = self.model.term_prob(_zs, action, task)

        # Compute losses
        reward_loss, value_loss, value_avg_loss = 0, 0, 0
        term_loss = 0
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
            s_abstract_loss = F.mse_loss(s_abstract_preds[t], s_abstract[t])
            value_avg_loss += (
                F.cross_entropy(v_avg[t], F.softmax(value_avg_targets[t], dim=-1))
                * self.cfg.rho**t
            )
            term_loss += F.binary_cross_entropy(
                term_probs[t], terminated[t].unsqueeze(-1).float()
            )
        consistency_loss *= 1 / self.cfg.horizon
        reward_loss *= 1 / self.cfg.horizon
        value_loss *= 1 / (self.cfg.horizon * self.cfg.num_q)
        value_avg_loss *= 1 / self.cfg.horizon
        term_loss *= 1 / self.cfg.horizon
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * value_loss
            + self.cfg.value_coef * value_avg_loss
            + self.cfg.term_coef * term_loss
            + s_abstract_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm
        )
        self.optim.step()

        # Update policy
        pi_loss = self.update_pi(zs.detach(), ks, s_starts.detach(), skills, task)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        value_avg_values = math.two_hot_inv(v_avg, self.cfg)
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
            "s_abstract_loss": float(s_abstract_loss.mean().item()),
            "q_values": float(torch.mean(
                torch.tensor([math.two_hot_inv(q.detach(), self.cfg).mean() for q in qs])
            ).item()),
            "value_avg_loss": float(value_avg_loss.mean().item()),
            "value_avg_values": value_avg_values.mean().item(),
        }
