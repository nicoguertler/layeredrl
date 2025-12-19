from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

from gymnasium.spaces import Space, Box
import numpy as np
from tensordict.tensordict import TensorDict
import torch


from . import Level
from .tdmpc2_level import TDMPC2Level, ProcessObsBuffer
from ..predictors import Predictor
from ..utils.misc import temp_eval_mode
from ..utils.normalization import RunningBatchNorm
from ..utils.schedules import ConstantSchedule
from ..tdmpc2.splates_tdmpc2 import SPlaTESTDMPC2


class RecomputeRewardBuffer(ProcessObsBuffer):
    """TD-MPC2 peplay buffer that recalculates the reward when sampling."""

    def __init__(
        self,
        parent_level: Level,
        control_interval: int,
        reward_batch_norm: bool = False,
        reward_bn_momentum: float = 0.01,
        reward_offset: float = 0.0,
        *args,
        **kwargs,
    ):
        """Initialize the buffer.

        Args:
            parent_level: The level owning this buffer.
            control_interval: For how many time steps one skill is executed.
            reward_batch_norm: Whether to use batch normalization on the reward.
            reward_bn_momentum: The momentum for the batch normalization.
            reward_offset: Offset to add to the reward after standardization.
        """
        super().__init__(*args, **kwargs)
        self.parent_level = parent_level
        self.control_interval = control_interval
        self.reward_offset = reward_offset
        if reward_batch_norm:
            self.reward_bn = RunningBatchNorm(
                1,
                momentum=reward_bn_momentum,
                dtype=torch.float32,
                device=self._device,
            )
        else:
            self.reward_bn = None

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size TxB.
        """
        obs, action, reward, task = super()._prepare_batch(td)
        terminated = td["terminated"][:-1]
        k = torch.fmod(
            self.control_interval - td["level_state"]["n_remaining_steps"],
            self.control_interval,
        )
        s_start = td["level_state"]["start_latent_state"]
        skill = td["level_input"]
        return self._to_device(obs, action, reward, k, s_start, skill, terminated, task)

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td, info = self._buffer.sample(return_info=True)
        td = td.view(-1, self.cfg.horizon + 1).permute(1, 0)
        td["obs"] = self._get_obs(
            td["mapped_env_obs"], td["level_input"], td["level_state"]
        )
        obs, action, reward, k, s_start, skill, terminated, task = self._prepare_batch(
            td
        )
        mapped_env_obs, level_input, level_state = self._to_device(
            td["mapped_env_obs"], td["level_input"], td["level_state"]
        )

        level_state_reshaped = {
            k: v[:-1].reshape(-1, *v[:-1].shape[2:]) for k, v in level_state.items()
        }

        new_reward, reward_info = self.parent_level.get_reward(
            mapped_env_obs[:-1].reshape(-1, *mapped_env_obs[:-1].shape[2:]),
            skill[:-1].reshape(-1, *level_input[:-1].shape[2:]),
            level_state_reshaped,
            action,
            mapped_env_obs[1:].reshape(-1, *mapped_env_obs[1:].shape[2:]),
            # Next level state is not used anyway
            level_state_reshaped,
            False,
            None,
            None,
        )
        if self.cfg.prioritized_sampling:
            indices = info["index"][0].view(-1, self.cfg.horizon + 1)[:, 1:].reshape(-1)
            self._buffer.update_priority(
                index=indices,
                priority=new_reward.exp().clamp(min=5e-2, max=5e1).nan_to_num(nan=0.05),
            )
        if self.reward_bn is not None:
            with torch.no_grad():
                new_reward = self.reward_bn(new_reward.unsqueeze(-1)).squeeze(-1)
        new_reward = new_reward + self.reward_offset
        new_reward = new_reward.view_as(reward)
        reward = new_reward

        s_abstract, _ = self.parent_level.parent_predictor.encode(
            mapped_env_obs.reshape(-1, obs.shape[-1])
        )
        s_abstract = s_abstract.view(*obs.shape[:2], -1)

        return obs, action, reward, k, s_start, skill, s_abstract, terminated, task

    def add(self, td):
        """Add an episode to the buffer."""
        num_eps, indices = super().add(td)
        if self.cfg.prioritized_sampling:
            # have to take into account that td contains reward going into the step
            # and not going out of the time step
            self._buffer.update_priority(
                index=indices[:-1],
                priority=td["reward"][1:].exp().clamp(min=5e-2, max=5e1),
            )
        return num_eps, indices

    def save(self, path: Path):
        """Save the buffer to the given path.

        Args:
            path: The path to save the buffer to."""
        super().save(path)
        if self.reward_bn is not None:
            # save the batch normalization state
            torch.save(self.reward_bn.state_dict(), path / "reward_bn.pt")

    def load(self, path: Path, td: TensorDict):
        """Load the buffer from the given path.

        Args:
            path: The path to load the buffer from.
            td: A TensorDict with the same keys as the buffer
                to infer the shape of the buffer from.
        """
        super().load(path, td)
        if self.reward_bn is not None:
            # Load the batch normalization state
            reward_bn_path = path / "reward_bn.pt"
            if reward_bn_path.exists():
                state_dict = torch.load(reward_bn_path)
                self.reward_bn.load_state_dict(state_dict)


class SPlaTESLevel(TDMPC2Level):
    """A level that learns skills using Stable Planning with Temporally Extended Skills (SPlaTES)."""

    def __init__(
        self,
        skill_space_dim: int,
        control_interval: int = 10,
        n_skill_samples: int = 100,
        reward_clipping_low: Union[float, Callable] = -50.0,
        reward_clipping_high: Union[float, Callable] = 50.0,
        potential_clipping_low: Union[float, Callable] = -50.0,
        reward_offset: float = 1.0,
        fixed_std: Optional[Union[float, torch.Tensor, Callable]] = 0.1,
        potential_diff_reward: bool = False,
        use_policy: bool = False,
        freeze_after: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the level.

        Args:
            skill_space_dim: The dimension of the skill space (or the number of skills in
                case of discrete skills).
            control_interval: How long to stay in control before control is returned to the
                level above.
            n_skill_samples: The number of skill samples to draw from skill prior for the
                'denominator' of the SPlaTES reward.
            reward_clipping_low: The value below which to clip the reward.
            reward_clipping_high: The value above which to clip the reward.
            potential_clipping_low: The value below which to clip the potential. This can improve
                exploration in the initial phase of learning (while clipping the reward too aggressively
                will make the agent go up and down the potential).
            reward_offset: Offset to add to the reward after standardization. This can be used to
                make the reward more positive in environments where the agent can terminate the episode
                early but is not supposed to do so.
            fixed_std: If not None, the standard deviation of the skill dynamics model is
                set to this value when calculating the intrinsic reward. The model itself
                is not changed. This allows for tuning the length scale of the mutual
                'repulsion' of the skills. Can also be a callable which maps the number of
                steps to a float in case of a schedule.
            potential_diff_reward: Whether to use the potential difference as reward instead of
                the potential itself.
            use_policy: Whether to use the policy instead of letting TD-MPC2 plan.
            freeze_after: After how many steps to freeze the skills. Optional.
            **kwargs: Keyword arguments for the TianshouLevel.
        """
        super().__init__(*args, **kwargs)
        self.latent_dimension = None
        self.context_dimension = None
        self.skill_space_dim = skill_space_dim
        self.skill_space = Box(low=-1.0, high=1.0, shape=(self.skill_space_dim,))
        self.control_interval = control_interval
        self.n_skill_samples = n_skill_samples
        self.reward_clipping_low = reward_clipping_low
        self.reward_clipping_high = reward_clipping_high
        self.potential_clipping_low = potential_clipping_low
        if not callable(reward_clipping_low):
            self.reward_clipping_low = ConstantSchedule(reward_clipping_low)
        if not callable(reward_clipping_high):
            self.reward_clipping_high = ConstantSchedule(reward_clipping_high)
        if not callable(potential_clipping_low):
            self.potential_clipping_low = ConstantSchedule(potential_clipping_low)
        self.reward_offset = reward_offset
        self._orig_fixed_std = fixed_std
        self.potential_diff_reward = potential_diff_reward
        self.use_policy = use_policy
        self.freeze_after = freeze_after

        self._rewards_sum = 0.0
        self._delta_norm_sum = 0.0
        self._n_skills_since_log = 0
        self.start_latent_state = None
        self.start_context = None
        self.policy_data = None

    def _create_buffer(self) -> None:
        """Create the replay buffer."""
        self.buffer = RecomputeRewardBuffer(
            cfg=self.tdmpc2_config,
            get_obs=self._get_tdmpc2_obs,
            parent_level=self,
            control_interval=self.control_interval,
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

    def initialize(
        self,
        env_obs_space: Space,
        action_space: Space,
        n_env_instances: int,
        parent_predictor: Predictor,
        env_obs_map: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mapped_env_obs_shape: Optional[int] = None,
        keep_params: bool = False,
    ) -> None:
        """Construct the level.

        Args:
            env_obs_space: The observation space of the environment.
            action_space: The action space of the level. If this is the lowest level,
                this is the action space of the environment. Otherwise, it is the
                input space of the level below this one.
            n_env_instances: The number of environment instances.
            parent_predictor: The predictor of the parent level (or None if there is no parent
                or the parent does not have a predictor).
            env_obs_map: A map that is applied to the environment observation.
                This can be used to implement information hiding and for moving a trained
                level from one environment to another with a different observation space.
                If None, the identity map is used.
            mapped_env_obs_shape: Shape of the output of env_obs_map. If None, the
                shape of the environment observation space is used.
            keep_params: Whether to keep the parameters of the level (e.g. the policy)
                when initializing. If False, the parameters are reset.
        """
        assert isinstance(
            env_obs_space, Box
        ), "Only Box observation spaces are supported."
        assert parent_predictor is not None, "SPlaTESLevel requires a parent predictor."
        self.latent_dimension = parent_predictor.encoder.latent_state_dim
        self.context_dimension = parent_predictor.encoder.context_dim
        self._action_low = torch.tensor(action_space.low, device=self.device)
        self._action_high = torch.tensor(action_space.high, device=self.device)
        # observation and action space for the TDMPC2 algorithm
        tdmpc2_obs_dim = self.get_tdmpc2_obs_shape(mapped_env_obs_shape, env_obs_space)
        self.tdmpc2_config.obs_shape = {self.tdmpc2_config["obs"]: tdmpc2_obs_dim}
        self.tdmpc2_config.action_dim = action_space.shape[0]
        self.tdmpc2_config.skill_dim = self.skill_space_dim
        self.tdmpc2_config.time_dim = 1
        self.tdmpc2_config.abstract_latent_dim = self.latent_dimension
        self.tdmpc2_config.control_interval = self.control_interval
        self._create_buffer()
        if not keep_params:
            self._tdmpc2 = SPlaTESTDMPC2(self.tdmpc2_config, device=self.device)
        self._last_tdmpc2_log_step = self.n_total_steps
        Level.initialize(
            self,
            env_obs_space,
            action_space,
            n_env_instances,
            parent_predictor,
            env_obs_map,
            mapped_env_obs_shape,
            keep_params,
        )
        if isinstance(self._orig_fixed_std, float):
            self.fixed_std = self._orig_fixed_std * torch.ones(
                self.latent_dimension, dtype=torch.float32, device=self.device
            )
        else:
            self.fixed_std = self._orig_fixed_std
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
        self.start_latent_state = torch.zeros(
            (self.n_env_instances, self.latent_dimension), device=self.device
        )
        self.start_context = torch.zeros(
            (self.n_env_instances, self.context_dimension), device=self.device
        )
        self.start_mapped_env_obs = torch.zeros(
            (self.n_env_instances,) + self.mapped_env_obs_shape, device=self.device
        )
        self._level_state_dims = self.get_level_state_dims()

    def get_level_state(
        self,
        mapped_env_obs: torch.Tensor,
        level_input: torch.Tensor,
        active_instances: torch.Tensor,
    ) -> Dict:
        """Compute the state of the level.

        This contains all information relevant for determining the level action except for
        the environment observation and the level input, which are handled separately.

        Args:
            mapped_env_obs: The mapped environment observation of the active instances.
            level_input: The input to this level for the active instances, i.e., the
                action from the level above.
            active_instances: The instances for which to return the level state.
        Returns:
            The level state.
        """
        level_state = {
            "n_remaining_steps": (
                self.control_interval - self.n_steps_in_control[active_instances]
            ).unsqueeze(-1),
            # SPlaTES level is conditioned on the latent state it found itself in when taking over control
            "start_latent_state": self.start_latent_state[active_instances],
            "start_context": self.start_context[active_instances],
        }
        return level_state

    def get_level_state_dims(self) -> Dict[str, int]:
        """Get the dimensions of the level state.

        Implement this in derived classes.

        Returns:
            A dictionary with the dimensions of each item in the level state.
        """
        dims = {
            "n_remaining_steps": 1,
            "start_latent_state": self.latent_dimension,
            "start_context": self.context_dimension,
        }
        return dims

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
        # give environment observation directly to TDMPC2, skill vector, start latent state,
        # etc. are treated separately
        return mapped_env_obs

    def reset(self):
        """Reset the level.

        Call once before the first episode, not between episodes."""
        super().reset()
        self._rewards_sum = 0.0
        self._splates_score = 0
        self._n_probs = 0
        self._last_log_step = self.n_total_steps

    def soft_reset(self):
        """Soft reset the hierarchy.

        This is called when manually resetting the (vector) environment. This will not
        affect things like warm up periods etc."""
        super().soft_reset()
        self._rewards_sum = 0.0
        self._splates_score = 0
        self._n_probs = 0
        self._last_log_step = self.n_total_steps

    def get_copy(self) -> "SPlaTESLevel":
        """Get a copy of the level.

        No parameters are copied, only the state of the level.

        Returns:
            A copy of the level.
        """
        level = super().get_copy()
        level.start_latent_state = torch.zeros(
            self.n_env_instances, self.latent_dimension, device=self.device
        )
        level.start_context = torch.zeros(
            self.n_env_instances, self.context_dimension, device=self.device
        )
        level.start_mapped_env_obs = torch.zeros(
            (self.n_env_instances,) + self.mapped_env_obs_shape, device=self.device
        )
        return level

    def sample_from_skill_prior(self, batch_size: torch.Size) -> torch.Tensor:
        """Sample from the skill prior.

        Args:
            batch_size: The batch size.
        Returns:
            The sampled skill vectors in a tensor.
        """
        return (
            2.0 * torch.rand(batch_size + (self.skill_space_dim,), device=self.device)
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

    def _get_potential(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        z: torch.Tensor,
        z_samples: torch.Tensor,
        next_state: torch.Tensor,
        fixed_std: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """Get potential from state, context, skill and next state.

        Args:
            state: The current state.
            context: The context.
            z: The skill.
            z_samples: The skill samples (for the 'denominator' of the potential).
            next_state: The next state.
            fixed_std: The fixed standard deviation of the skill dynamics model.
        Returns:
            The reward and the reward info dict."""
        # probability of transition under current skill
        log_prob_z, _, z_info = self.parent_predictor.get_standardized_log_prob(
            state, context, z, next_state, fixed_std
        )

        n_repetitions = self.n_skill_samples
        state_samples_shape = state.shape[:-1] + (n_repetitions,) + state.shape[-1:]
        state_samples = state[..., None, :].expand(*state_samples_shape)
        next_state_samples = next_state[..., None, :].expand(*state_samples_shape)
        context_samples = context[..., None, :].expand(
            *state_samples_shape[:-1], context.shape[-1]
        )
        # reshape to merge batch and sample dimension
        state_samples = state_samples.reshape(-1, *state.shape[-1:])
        context_samples = context_samples.reshape(-1, *context.shape[-1:])
        next_state_samples = next_state_samples.reshape(-1, *state.shape[-1:])
        z_samples = z_samples.reshape(-1, *z.shape[-1:])
        # probability of transition under sampled skills
        log_prob_z_samples, _, _ = self.parent_predictor.get_standardized_log_prob(
            state_samples,
            context_samples,
            z_samples,
            next_state_samples,
            fixed_std,
        )
        log_prob_z_samples = log_prob_z_samples.reshape(
            *state.shape[:-1], n_repetitions
        )
        # max log prob for logsumexp trick
        max_log_prob = log_prob_z_samples.max(dim=-1).values
        max_log_prob = torch.max(max_log_prob, log_prob_z)
        # add actual z to pool of sampled zs
        log_prob_z_samples = (log_prob_z_samples - max_log_prob[..., None]).exp().sum(
            dim=-1
        ) + (log_prob_z - max_log_prob).exp()
        log_prob_z_samples = log_prob_z_samples.log() + max_log_prob
        log_prob_z_samples -= torch.log(
            torch.tensor(
                self.n_skill_samples + 1, device=self.device, dtype=torch.double
            )
        )
        # reward
        r = log_prob_z - log_prob_z_samples
        r = r.nan_to_num(
            nan=self.reward_clipping_low(self.n_total_steps),
            posinf=self.reward_clipping_high(self.n_total_steps),
            neginf=self.reward_clipping_low(self.n_total_steps),
        )
        r = r.clamp(min=self.potential_clipping_low(self.n_total_steps)).float()
        info = {
            "splates_score": r,
            "log_prob_z": log_prob_z,
            "log_prob_z_samples": log_prob_z_samples,
        }
        return r, info, z_info

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
        """Get the reward for the given batch of transitions (difference of mutual information estimates).

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
        # switch to eval mode to disable changing batch normalization etc.
        with torch.no_grad(), temp_eval_mode(self.parent_predictor):
            state = level_state["start_latent_state"]
            current_state, _ = self.parent_predictor.encode(mapped_env_obs)
            context = level_state["start_context"]
            z = level_input
            if use_next_level_state:
                next_state = next_level_state["start_latent_state"]
            else:
                next_state, _ = self.parent_predictor.encode(next_mapped_env_obs)

            # use same samples for potential before and after transition
            # to avoid influence of noise on potential difference
            z_samples = self.sample_from_skill_prior(
                state.shape[:-1] + (self.n_skill_samples,)
            )

            # calculate potential with current latent state as next state
            if self.potential_diff_reward:
                r_current, _, _ = self._get_potential(
                    state=state,
                    context=context,
                    z=z,
                    z_samples=z_samples,
                    next_state=current_state if self.potential_diff_reward else state,
                    fixed_std=fixed_std,
                )
            # calculate potential with next latent state as next state
            r_next, info_next, z_info = self._get_potential(
                state=state,
                context=context,
                z=z,
                z_samples=z_samples,
                next_state=next_state,
                fixed_std=fixed_std,
            )

            last_step = (level_state["n_remaining_steps"] == 1).squeeze(-1)
            if self.potential_diff_reward:
                r = r_next - r_current
            else:
                r = torch.zeros_like(r_next)
                r[last_step] = r_next[last_step]

            r = r.clamp(
                self.reward_clipping_low(self.n_total_steps),
                self.reward_clipping_high(self.n_total_steps),
            ).float()
            info = {
                "splates_score": None,
                "log_prob_z": None,
                "log_prob_z_samples": None,
                "current_state": current_state,
                "next_state": next_state,
            }
        if last_step.sum() > 0:
            # log probabilities and splates score only for last step
            info.update(info_next)
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
            level_input_info: Additional information about the level input.
            active_instances: In which of the environment instances the level is active.
                env_obs and level_input correspond to these instances.
        Returns:
            The action (also with a batch dimension).
            An info dict containing additional information about the action.
        """
        taking_over_control = self.n_steps_in_control[active_instances] == 0
        # figure out which instances are active and taking over control
        indices = torch.nonzero(active_instances)[taking_over_control].squeeze(-1)
        with temp_eval_mode(self.parent_predictor):
            self.start_latent_state[indices], self.start_context[indices] = (
                self.parent_predictor.encode(mapped_env_obs[taking_over_control])
            )
        # store the mapped environment observation at the start of the skill execution
        self.start_mapped_env_obs[indices] = mapped_env_obs[taking_over_control]

        if self.n_total_steps > self.warm_up_steps:
            level_state = self.get_level_state(
                mapped_env_obs, level_input, active_instances
            )
            # process environment observation, level input and level state
            tdmpc2_obs = self._get_tdmpc2_obs(
                mapped_env_obs[active_instances], level_input, level_state
            )
            # use tdmpc2 and condition on k, s_start, skil
            action = self._tdmpc2.act(
                tdmpc2_obs,
                k=self.control_interval - level_state["n_remaining_steps"],
                s_start=level_state["start_latent_state"],
                skill=level_input,
                t0=self.n_steps_in_control[active_instances] == 0,
                use_policy=self.use_policy,
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
        # return control if the control interval is over
        control_expired = (
            self.n_steps_in_control[active_instances] + 1 >= self.control_interval
        )
        self.return_control = control_expired
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

            if self._reward_info:
                if self._reward_info["splates_score"] is not None:
                    self._n_probs += self._reward_info["splates_score"].numel()
                    self._splates_score += self._reward_info["splates_score"].sum()

            steps_elapsed_since_log = self.n_total_steps - self._last_log_step
            if self.writer is not None and steps_elapsed_since_log > self.log_interval:
                # logging
                self._last_log_step = self.n_total_steps
                mean_reward = (
                    self._rewards_sum
                    / steps_elapsed_since_log
                    * self.reward_calc_interval
                )
                self.writer.add_scalar(
                    "splates-level/r", mean_reward, self.n_total_env_steps
                )
                if self.buffer.reward_bn is not None:
                    self.writer.add_scalar(
                        "splates-level/reward-batch-norm_std",
                        torch.sqrt(self.buffer.reward_bn.var),
                    )
                    self.writer.add_scalar(
                        "splates-level/reward-batch-norm_mean",
                        self.buffer.reward_bn.mean,
                    )
                if self._n_probs > 0:
                    self.writer.add_scalar(
                        "splates-level/splates_score",
                        self._splates_score / self._n_probs,
                        self.n_total_env_steps,
                    )
                self._n_probs = 0
                self._splates_score = 0
                self._rewards_sum = 0.0
                self._delta_norm_sum = 0.0

        return self.return_control

    def learn(self) -> None:
        """Learn from the collected (semi-MDP) transitions."""
        if self.freeze_after is None or self.n_total_env_steps < self.freeze_after:
            super().learn()
