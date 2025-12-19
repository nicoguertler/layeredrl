from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

from gymnasium.spaces import Space, Box, Discrete
import numpy as np
from tensordict.tensordict import TensorDict
import torch

from . import Level
from .tdmpc2_level import TDMPC2Level
from ..tdmpc2.common.buffer import Buffer
from ..utils.normalization import RunningBatchNorm
from ..utils.schedules import ConstantSchedule
from ..utils.misc import temp_eval_mode


class DADSBuffer(Buffer):
    """TD-MPC2 buffer for use with DADS level."""

    def __init__(
        self,
        cfg: Dict,
        get_obs: Callable,
        parent_level: Level,
        reward_func: Optional[Callable] = None,
        reward_batch_norm: bool = False,
        reward_bn_momentum: float = 0.01,
        reward_offset: float = 0.0,
    ):
        """Initialize the buffer.

        Args:
            cfg: The config of the buffer.
            get_obs: A function that concatenates the observations.
            parent_level: The level owning this buffer.
            reward_func: A function that computes the reward from the achieved and desired goal.
            reward_batch_norm: Whether to use batch normalization on the reward.
            reward_bn_momentum: The momentum for the batch normalization.
            reward_offset: Offset to add to the reward after standardization.
        """
        super().__init__(cfg)
        self._get_obs = get_obs
        self.reward_func = reward_func
        self.parent_level = parent_level
        if reward_batch_norm:
            self.reward_bn = RunningBatchNorm(
                1,
                momentum=reward_bn_momentum,
                dtype=torch.float32,
                device=self._device,
            )
        else:
            self.reward_bn = None
        self.reward_offset = reward_offset

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td, info = self._buffer.sample(return_info=True)

        td = td.view(-1, self.cfg.horizon + 1).permute(1, 0)

        mapped_env_obs = td["mapped_env_obs"]
        level_input = td["level_input"]
        obs = self._get_obs(td["mapped_env_obs"], td["level_input"], td["level_state"])
        action = td["action"][1:]
        reward = td["reward"][1:].unsqueeze(-1)
        task = td["task"][0] if "task" in td.keys() else None
        terminated = td["terminated"][:-1]
        skill = td["level_input"]

        new_reward, reward_info = self.parent_level.get_reward(
            mapped_env_obs[:-1].reshape(-1, *mapped_env_obs[:-1].shape[2:]),
            skill[:-1].reshape(-1, *level_input[:-1].shape[2:]),
            # level state not used
            None,
            action,
            mapped_env_obs[1:].reshape(-1, *mapped_env_obs[1:].shape[2:]),
            # next level state not used
            None,
            False,
            None,
            None,
        )
        if self.reward_bn is not None:
            with torch.no_grad():
                new_reward = self.reward_bn(new_reward.unsqueeze(-1)).squeeze(-1)
        new_reward = new_reward + self.reward_offset
        new_reward = new_reward.view_as(reward)
        reward = new_reward

        return self._to_device(obs, skill, action, reward, terminated, task)

    def save(self, path: Path):
        """Save the buffer to the given path.

        Args:
            path: The path to save the buffer to."""
        assert hasattr(self, "_buffer"), "No buffer to save."
        self._buffer.dumps(path)

    def load(self, path: Path, td: TensorDict):
        """Load the buffer from the given path.

        Args:
            path: The path to load the buffer from.
            td: A TensorDict with the same keys as the buffer
                to infer the shape of the buffer from.
        """
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.loads(path)
        unique_ep_ids = self._buffer.storage["episode"].unique()
        self._num_eps = unique_ep_ids.shape[0]


class DADSLevel(TDMPC2Level):
    """A level that learns skills using DADS.

    For the original DADS paper, see https://arxiv.org/abs/1907.01657

    While the original DADS paper uses SAC, this implementation uses TD-MPC2
    to learn the skills.
    """

    def __init__(
        self,
        skill_space_dim: int,
        discrete_skills: bool = False,
        control_interval: int = 1,
        termination_probability: float = 0.0,
        n_skill_samples: int = 100,
        reward_clipping_low: Union[float, Callable] = -50.0,
        reward_clipping_high: Union[float, Callable] = 50.0,
        reward_offset: float = 1.0,
        warm_up_steps: int = 2000,
        reward_scale: float = 1.0,
        bootstrap: bool = False,
        log_interval: int = 1000,
        fixed_std: Optional[float] = 1.0,
        scale_rew_with_std: bool = False,
        **kwargs
    ) -> None:
        """Initialize the level.

        Args:
            skill_space_dim: The dimension of the skill space (or the number of skills in
                case of discrete skills).
            discrete_skills: Whether the skills are discrete or continuous. Only continuous skills are supported
                for now.
            control_interval: How long to stay in control before control is returned to the
                level above.
            termination_probability: The probability of terminating the skill after a transition
                before the control interval has elapsed.
            n_skill_samples: The number of skill samples to draw from skill prior for the
                'denominator' of the DADS reward.
            reward_clipping_low: The value below which to clip the reward.
            reward_clipping_high: The value above which to clip the reward.
            reward_offset: Offset to add to the reward after standardization. This can be used to
                make the reward more positive in environments where the agent can terminate the episode
                early but is not supposed to do so.
            warm_up_steps: The number of steps to collect before learning starts.
            reward_scale: Factor to multiply onto rewards.
            bootstrap: Whether to bootstrap when the skill vector does not change
                in a transition. This might be necessary for problems that do not
                allow differentiation in the transitions within one time step.
            log_interval: The interval in which to log statistics.
            fixed_std: If not None, the standard deviation of the skill dynamics model is
                set to this value when calculating the intrinsic reward. The model itself
                is not changed. This allows for tuning the length scale of the mutual
                'repulsion' of the skills.
            scale_rew_with_std: Whether to scale the reward with the squared standard deviation
                of the model. This is useful for approximately fixing the reward scale when
                tuning fixed_std and not messing up the exploration bonuses.
            **kwargs: Keyword arguments for the TianshouLevel.
        """
        # use goal_based=True as skill and goal are interchangeable in GoalBasedTDMPC2
        super().__init__(
            goal_based=True, use_her=False, goal_dim=skill_space_dim, **kwargs
        )
        self.skill_space_dim = skill_space_dim
        assert (
            not discrete_skills
        ), "Discrete skills are not supported in DADSLevel at the moment."
        self.discrete_skills = discrete_skills
        if discrete_skills:
            self.skill_space = Discrete(self.skill_space_dim)
        else:
            self.skill_space = Box(low=-1.0, high=1.0, shape=(self.skill_space_dim,))
        self.control_interval = control_interval
        self.termination_probability = termination_probability
        self.n_skill_samples = n_skill_samples
        self.reward_clipping_low = reward_clipping_low
        self.reward_clipping_high = reward_clipping_high
        if not callable(reward_clipping_low):
            self.reward_clipping_low = ConstantSchedule(reward_clipping_low)
        if not callable(reward_clipping_high):
            self.reward_clipping_high = ConstantSchedule(reward_clipping_high)
        self.reward_offset = reward_offset
        self.warm_up_steps = warm_up_steps
        self.reward_scale = reward_scale
        self.bootstrap = bootstrap
        self.log_interval = log_interval
        self._orig_fixed_std = fixed_std
        self.scale_rew_with_std = scale_rew_with_std
        self._warm_up = True
        self._rewards_sum = 0.0

    def _create_buffer(self) -> None:
        """Create the replay buffer."""
        # This replay buffer recalculates the DADS reward when sampling.
        self.buffer = DADSBuffer(
            cfg=self.tdmpc2_config,
            get_obs=self._get_tdmpc2_obs,
            parent_level=self,
            reward_batch_norm=True,
            reward_bn_momentum=0.005,
            reward_offset=self.reward_offset,
        )

    def get_tdmpc2_obs_shape(
        self, mapped_env_obs_shape: Tuple[int, ...], env_obs_space: Box
    ) -> int:
        """Get the shape of the observation for the TDMP2 algorithm."""
        actual_mapped_env_obs_dim = (
            mapped_env_obs_shape[0]
            if mapped_env_obs_shape is not None
            else env_obs_space.shape[0]
        )
        return (actual_mapped_env_obs_dim,)

    def initialize(self, parent_predictor, keep_params=False, *args, **kwargs) -> None:
        """Initialize the level.

        Args:
            keep_params: Whether to keep the parameters of the level (e.g. the policy)
                when initializing. If False, the parameters are reset.
            *args: Arguments for the TianshouLevel.
            **kwargs: Keyword arguments for the TianshouLevel.
        """
        super().initialize(
            *args, parent_predictor=parent_predictor, keep_params=keep_params, **kwargs
        )
        self.latent_dimension = parent_predictor.encoder.latent_state_dim
        if isinstance(self._orig_fixed_std, float):
            self.fixed_std = self._orig_fixed_std * torch.ones(
                self.latent_dimension, dtype=torch.float32, device=self.device
            )
        else:
            self.fixed_std = self._orig_fixed_std
        if self.latent_dimension is None:
            self.latent_dimension = self.mapped_env_obs_shape[-1]
        self.latent_space = Box(
            low=-np.inf, high=np.inf, shape=(self.latent_dimension,)
        )
        self._last_log_step = 0

    def set_n_env_instances(self, n_env_instances: int) -> None:
        """Set the number of environment instances.

        Args:
            n_env_instances: The number of environment instances.
        """
        super().set_n_env_instances(n_env_instances)
        self._level_state_dims = self.get_level_state_dims()

    def get_input_space(self) -> Space:
        """Get the input space of level (the skill space).

        Returns:
            The input space (skill space).
        """
        return self.skill_space

    def _get_tdmpc2_obs(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: torch.Tensor,
        level_state: torch.Tensor,
    ):
        """Get the observation for the TDMPC2 algorithm.

        Args:
            mapped_env_obs: The environment observation after the self.env_obs_map has been
                applied. Note that the observation has a batch dimension (for multiple
                environment instances). Only for active instances.
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            level_state: The state of the level."""
        # give environment observation directly to TDMPC2
        return mapped_env_obs

    def reset(self):
        """Reset the level.

        Call once before the first episode, not between episodes."""
        super().reset()
        self._warm_up = True
        self._rewards_sum = 0.0

    def soft_reset(self):
        """Soft reset the hierarchy.

        This is called when manually resetting the (vector) environment. This will not
        affect things like warm up periods etc."""
        super().soft_reset()
        self._rewards_sum = 0.0
        self._last_log_step = self.n_total_steps

    def get_copy(self) -> "DADSLevel":
        """Get a copy of the level.

        No parameters are copied, only the state of the level.

        Returns:
            A copy of the level.
        """
        level = super().get_copy()
        return level

    def sample_from_skill_prior(self, batch_size: torch.Size) -> torch.Tensor:
        """Sample from the skill prior.

        Args:
            batch_size: The batch size.
        Returns:
            The sampled skill vectors in a tensor.
        """
        if self.discrete_skills:
            # enumerate all skills in discrete case
            indices = torch.arange(0, self.skill_space_dim, device=self.device)
            indices = indices.expand(batch_size[:-1] + (self.skill_space_dim,))
            return torch.nn.functional.one_hot(indices, self.skill_space_dim).float()
        else:
            return (
                2.0
                * torch.rand(batch_size + (self.skill_space_dim,), device=self.device)
                - 1.0
            )

    def get_fixed_std(self) -> torch.Tensor:
        """Get the fixed standard deviation of the skill dynamics model.

        Can depend on the total number of steps taken in the environment in case of a schedule.
        """
        # calculate fixed std (in case of a schedule)
        if callable(self._orig_fixed_std):
            return self._orig_fixed_std(self.n_total_steps) * torch.ones(
                self.latent_dimension, device=self.device
            )
        else:
            return self.fixed_std

    def get_reward(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_state: Dict,
        action: torch.Tensor,
        next_mapped_env_obs: torch.Tensor,
        next_level_state: Dict,
        terminated: bool,
        cum_env_reward: torch.Tensor,
        elapsed_env_steps: torch.Tensor,
        use_next_level_state: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """Get the reward for the given batch of transitions.

        Note that everything has a batch dimension. Also note that there is no next_level_input
        because the level input doeos not change while the level is in control.

        Args:
            mapped_env_obs: The mapped environment observation.
            level_input: The input to this level.
            level_state: The state of the level.
            action: The action this level took.
            next_mapped_env_obs: The next mapped environment observation.
            next_level_state: The next state of the level.
            terminated: Whether the episode terminated.
            cum_env_reward: The sum of the environment rewards that were received
                during the transition.
            elapsed_env_steps: The number of environment steps that elapsed during
                the transition.
            use_next_level_state: Whether to use the next level state instead of next_mapped_env_obs.
                This is useful for visualization purposes where the next state is set manually.
        Returns:
            The reward and the reward info dict.
        """
        fixed_std = self.get_fixed_std()
        with torch.no_grad(), temp_eval_mode(self.parent_predictor):
            state, context = self.parent_predictor.encode(mapped_env_obs)
            z = level_input
            if use_next_level_state:
                next_state = next_level_state["start_latent_state"]
            else:
                next_state, _ = self.parent_predictor.encode(next_mapped_env_obs)

            # probability of transition under current skill
            log_prob_z, _, z_info = self.parent_predictor.get_standardized_log_prob(
                state, context, z, next_state, fixed_std
            )

            z_samples = self.sample_from_skill_prior(
                state.shape[:-1] + (self.n_skill_samples,)
            )
            n_repetitions = (
                self.skill_space_dim if self.discrete_skills else self.n_skill_samples
            )
            state_samples_shape = state.shape[:-1] + (n_repetitions,) + state.shape[-1:]
            state_samples = state[..., None, :].expand(*state_samples_shape)
            context_samples = context[..., None, :].expand(
                *state_samples_shape[:-1], context.shape[-1]
            )
            next_state_samples = next_state[..., None, :].expand(*state_samples_shape)
            # reshape to merge batch and sample dimension
            state_samples = state_samples.reshape(-1, *state.shape[-1:])
            context_samples = context_samples.reshape(-1, *context.shape[-1:])
            next_state_samples = next_state_samples.reshape(-1, *state.shape[-1:])
            z_samples = z_samples.reshape(-1, *z.shape[-1:])
            # probability of transition under sampled skills
            log_prob_z_samples, _, z_samples_info = (
                self.parent_predictor.get_standardized_log_prob(
                    state_samples,
                    context_samples,
                    z_samples,
                    next_state_samples,
                    fixed_std,
                )
            )
            log_prob_z_samples = log_prob_z_samples.reshape(
                *state.shape[:-1], n_repetitions
            )
            # max log prob for logsumexp trick
            max_log_prob = log_prob_z_samples.max(dim=-1).values
            max_log_prob = torch.max(max_log_prob, log_prob_z)
            # add actual z to pool of sampled zs
            log_prob_z_samples = (
                log_prob_z_samples - max_log_prob[..., None]
            ).exp().sum(dim=-1) + (log_prob_z - max_log_prob).exp()
            log_prob_z_samples = log_prob_z_samples.log() + max_log_prob
            log_prob_z_samples -= torch.log(
                torch.tensor(
                    self.n_skill_samples + 1, device=self.device, dtype=torch.double
                )
            )
            r = log_prob_z - log_prob_z_samples
            r = self.reward_scale * r
            if self.scale_rew_with_std and fixed_std is not None:
                r = r * fixed_std**2
            r = r.nan_to_num(
                nan=self.reward_clipping_low(self.n_total_steps),
                posinf=self.reward_clipping_high(self.n_total_steps),
                neginf=self.reward_clipping_low(self.n_total_steps),
            )
            r = r.clamp(
                self.reward_clipping_low(self.n_total_steps),
                self.reward_clipping_high(self.n_total_steps),
            ).float()
            info = {
                "log_prob_z": log_prob_z,
                "log_prob_z_samples": log_prob_z_samples,
            }
            return r, info

    def get_action(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        level_input_info: Optional[Dict],
        active_instances: torch.Tensor = torch.tensor([True], dtype=torch.bool),
    ) -> torch.Tensor:
        """Get an action for the given observation.

        Note that only the action for the active instances is returned.

        Call this at the beginning of the implementation of get_action in derived classes.

        Args:
            mapped_env_obs: The environment observation after the self.env_obs_map has been
                applied. Note that the observation has a batch dimension (for multiple
                environment instances).
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            active_instances: In which of the environment instances the level is active.
                env_obs and level_input correspond to these instances.
        Returns:
            The action (also with a batch dimension).
            An info dict containing additional information about the action.
        """
        if self.n_total_steps > self.warm_up_steps:
            # concatenate environment observation, level input and level state
            tdmpc2_obs = self._get_tdmpc2_obs(
                mapped_env_obs[active_instances], None, None
            )
            # use tdmpc2
            action = self._tdmpc2.act(
                tdmpc2_obs,
                g=level_input,
                t0=self.n_steps_in_control[active_instances] == 0,
                eval_mode=not self.training,
            )
            action = self._action_low + 0.5 * (action + 1) * (
                self._action_high - self._action_low
            )
        else:
            action = self.sample_from_action_space(active_instances.sum())
        return action, dict()

    def process_transition(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: Optional[torch.Tensor],
        action: torch.Tensor,
        next_mapped_env_obs: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        active_instances: torch.Tensor,
    ) -> bool:
        """Process transition and check whether level would like to return
        control to the level above.

        This usually involves adding the transition to the replay buffer and possibly
        preprocessing it.

        Note that everything has a batch dimension.

        Args:
            mapped_env_obs: The mapped environment observation for the active instances.
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            action: The action that was taken by the level.
            next_mapped_env_obs: The next mapped environment observation for the active instances.
            terminated: Whether the episode terminated for the active instances.
            truncated: Whether the episode was truncated for the active instances.
            active_instances: In which of the environment instances the level is active.
                next_obs and terminated correspond to these instances.
        Returns:
            Whether the level is done, i.e. whether it hands control back to
            the level above.
        """
        # return control if either the episode terminated or the control interval is over
        control_expired = (
            self.n_steps_in_control[active_instances] + 1 >= self.control_interval
        )
        stochastic_term = (
            torch.rand(active_instances.shape, device=self.device)
            < self.termination_probability
        )
        self.return_control = torch.logical_or(control_expired, stochastic_term)
        super().process_transition(
            mapped_env_obs,
            level_input,
            action,
            next_mapped_env_obs,
            terminated,
            truncated,
            active_instances,
        )

        if self.training:
            # for logging later
            self._rewards_sum += self._newest_transition["reward"].sum()

            steps_elapsed_since_log = self.n_total_steps - self._last_log_step
            if self.writer is not None and steps_elapsed_since_log > self.log_interval:
                # logging
                self._last_log_step = self.n_total_steps
                # log statistics of new transition
                mean_reward = (
                    self._rewards_sum
                    / steps_elapsed_since_log
                    * self.reward_calc_interval
                )
                self.writer.add_scalar(
                    "dads-level/reward", mean_reward, self.n_total_env_steps
                )
                self._rewards_sum = 0.0
        return self.return_control
