from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import gymnasium as gym
from tianshou.data import Batch

from ..hierarchies import Hierarchy
from ..utils.misc import to_numpy, copy_torch_or_numpy, temp_eval_mode
from ..utils.loggers import VideoLogger


class Collector:
    def __init__(
        self,
        hierarchy: Hierarchy,
        env: gym.Env,
        test_env: Optional[gym.Env] = None,
        device=torch.device("cpu"),
        writer: Optional[SummaryWriter] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: Optional[int] = None,
    ):
        """Initialize the collector.

        Args:
            hierarchy: The hierarchy to collect data with.
            env: The environment to collect data/train in.
            test_env: The environment to test in.
            device: The device to use.
            writer: The TensorBoard writer to use for logging. If None, no logging is done.
            checkpoint_dir: The directory to save checkpoints to. If None, no checkpoints are saved.
            checkpoint_interval: The interval in steps between checkpoints. If None, only the final checkpoint is saved.
        """
        self.hierarchy = hierarchy
        self.env = self._to_vec_env(env)
        self.test_env = self._to_vec_env(test_env) if test_env is not None else None
        self.n_envs = self.env.num_envs
        self.device = device
        self.writer = writer
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.success_key = None
        self._single_test_stats = None

    def save_checkpoint(self, t: int, n_steps: int):
        """Save a checkpoint of the hierarchy.

        Args:
            t: The current step.
        """
        if self.checkpoint_dir is not None:
            if (t == n_steps - 1) or (
                self.checkpoint_interval is not None
                and t % self.checkpoint_interval == 0
            ):
                new_checkpoint_dir = self.checkpoint_dir / f"checkpoint_{t}"
                new_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.hierarchy.save(new_checkpoint_dir)

    def _to_vec_env(self, env: gym.Env) -> gym.vector.AsyncVectorEnv:
        """Convert an environment to a vector environment.

        If the environment is already a vector environment, it is returned as is.

        Args:
            env: The environment to convert.
        Returns:
            A vector environment.
        """
        if isinstance(env, gym.vector.VectorEnv):
            env = env
        else:
            # Convert single environment to a vector environment
            env = gym.vector.AsyncVectorEnv(
                [lambda: env], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
            )
        return env

    def reset(self, seed=None):
        """Reset the collector.

        Call this at the beginning of the session."""
        self.obs, self.info = self.env.reset(seed=seed)
        # Have to step once as some environments don't provide
        # success key in info after reset.
        _, _, _, _, step_info = self.env.step(self.env.action_space.sample())
        if "success" in step_info:
            self.success_key = "success"
        elif "is_success" in step_info:
            self.success_key = "is_success"
        else:
            self.success_key = None
        # reset again
        self.obs, self.info = self.env.reset(seed=seed)
        self.terminated = False
        self.truncated = False
        self.hierarchy.reset()

    def test(
        self,
        t: int,
        test_hierarchy: Hierarchy,
        n_steps: int,
        env_expects_numpy: bool = True,
        video_logger: Optional[VideoLogger] = None,
    ) -> dict:
        """Test the hierarchy in the environment.

        Note: This resets the test environment and the test hierarchy.

        Args:
            t: The current training step.
            n_steps: The number of steps to test the hierarchy.
            env_expects_numpy: Whether the environment expects numpy arrays as input.
            video_logger: A VideoLogger object to log videos of the rollouts.
        Returns:
            The statistics of the test run.
        """
        obs, _ = self.test_env.reset()
        # this (soft) resets the internal state of the hierarchy,
        test_hierarchy.soft_reset()

        rewards = np.zeros(self.n_envs)
        n_episodes = np.zeros(self.n_envs, dtype=int)
        success_in_ep = np.zeros(self.n_envs, dtype=bool)
        success_in_ep_cum = np.zeros(self.n_envs, dtype=int)
        success_frac = np.zeros(self.n_envs)

        with temp_eval_mode(test_hierarchy):
            for _ in range(n_steps):
                with torch.no_grad():
                    action = test_hierarchy.get_action(obs)
                if env_expects_numpy:
                    action = to_numpy(action)
                obs_next, reward, terminated, truncated, info = self.test_env.step(
                    action
                )
                rewards += reward
                if "_final_observation" in info:
                    # provide final observation of episode to hierarchy
                    obs_next_transition = copy_torch_or_numpy(obs_next)
                    for i, env_final_obs in enumerate(info["final_observation"]):
                        if env_final_obs is not None:
                            obs_next_transition[i] = env_final_obs
                else:
                    obs_next_transition = obs_next

                if video_logger is not None:
                    video_logger.log_frame(self.test_env, terminated, truncated)

                test_hierarchy.process_transition(
                    obs_next=obs_next_transition,
                    rew=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
                done = np.logical_or(terminated, truncated)
                n_episodes[done] += 1
                if self.success_key is not None:
                    if self.success_key in info:
                        success_in_ep = np.logical_or(
                            info[self.success_key], success_in_ep
                        )
                        success_frac += info[self.success_key].astype(np.float64)
                    if done.any():
                        success_in_ep_cum[done] += success_in_ep[done].astype(int)
                        success_in_ep[done] = False
                obs = obs_next
        mean_reward = rewards.mean() / n_steps
        mean_return = rewards.mean() / np.maximum(n_episodes, 1).mean()
        mean_success_in_ep = success_in_ep_cum.mean() / np.maximum(n_episodes, 1).mean()
        mean_success_frac = success_frac.mean() / n_steps
        if self.writer is not None:
            # logging
            self.writer.add_scalar("test_reward", mean_reward, t * self.n_envs)
            self.writer.add_scalar("test_return", mean_return, t * self.n_envs)
            self.writer.add_scalar(
                "test_success_frac", mean_success_frac, t * self.n_envs
            )
            self.writer.add_scalar(
                "test_success_in_ep", mean_success_in_ep, t * self.n_envs
            )
        stats = {
            "test_reward": mean_reward,
            "test_return": mean_return,
            "test_success_frac": mean_success_frac,
            "test_success_in_ep": mean_success_in_ep,
        }
        return stats

    def collect(
        self,
        n_steps: int,
        env_expects_numpy: bool = True,
        record_transitions: bool = False,
        learn: bool = False,
        n_steps_start: int = 0,
        log_interval: int = 100,
        test_interval: Optional[int] = None,
        n_test_steps: int = 1000,
        verbose: bool = False,
        post_step_callback: Optional[Callable] = None,
        video_logger: Optional[VideoLogger] = None,
    ) -> Union[Tuple, Batch]:
        """Collect transitions from the environment with the hierarchical policy.

        This collects different transitions on every level of the hierarchy as
        the higher levels see semi MDPs.

        Args:
            n_steps: The number of steps to collect in each environment instance. The total number of
            collected steps is therefore n_steps * n_envs.
            env_expects_numpy: Whether the environment expects numpy arrays as input.
            record_transitions: Whether to record the environment transitions and return them in a batch.
                Note that the first dimension of the batch corresponds to the step, not the number of
                environment instances.
            learn: Whether to learn after each step.
            n_steps_start: The number of steps that have already been collected. This is useful  for for
                resuming an experiment.
            log_interval: The interval in steps between logging.
            test_interval: The interval in vector environment steps between testing the hierarchy. If None,
                no testing is done.
            n_test_steps: The number of vector environment steps to test the hierarchy for at each test interval.
            verbose: Whether to print progress.
            post_step_callback: A callback function that is called after each step. The callback
                function should take the current step and the next observation as an argument.
            video_logger: A VideoLogger object to log videos of the rollouts.
        Returns:
            The statistics of the rollouts and (if record_transitions is Ture) the collected transitions in
            a Batch object.
        """
        n_episodes = np.zeros(self.n_envs, dtype=int)
        episode_lengths = np.zeros(self.n_envs, dtype=int)
        test_stats = {
            "test_reward": [],
            "test_return": [],
            "test_success_frac": [],
            "test_success_in_ep": [],
        }
        cum_reward = np.zeros(self.n_envs)
        episode_return = np.zeros(self.n_envs)
        reward_since_last_log = 0.0
        is_success_since_last_log = 0.0
        n_steps_diff = n_steps - n_steps_start
        if record_transitions:
            obs_seq = np.empty((n_steps_diff + 1,) + self.obs.shape)
            obs_seq[0] = self.obs
            act_seq = np.empty((n_steps_diff,) + self.env.action_space.shape)
            rew_seq = np.empty((n_steps_diff, self.n_envs))
            term_seq = np.empty((n_steps_diff, self.n_envs), dtype=bool)
            trunc_seq = np.empty((n_steps_diff, self.n_envs), dtype=bool)
        if verbose:
            step_range = trange(n_steps_start, n_steps)
        else:
            step_range = range(n_steps_start, n_steps)

        # iterate over environment steps
        for t in step_range:
            with torch.no_grad():
                action = self.hierarchy.get_action(self.obs)
            if env_expects_numpy:
                action = to_numpy(action)

            obs_next, reward, terminated, truncated, self.info = self.env.step(action)
            episode_lengths += 1
            done = np.logical_or(terminated, truncated)

            reward_since_last_log += reward.mean()
            episode_return += reward
            if self.success_key is not None and self.success_key in self.info:
                is_success_since_last_log += self.info[self.success_key].mean()

            if video_logger is not None:
                video_logger.log_frame(self.env, terminated, truncated)

            if post_step_callback is not None:
                post_step_callback(t, obs_next)

            if self.writer is not None:
                # logging
                if t % log_interval == 0:
                    self.writer.add_scalar(
                        "reward", reward_since_last_log / log_interval, t * self.n_envs
                    )
                    self.writer.add_scalar("vec_steps", t * self.n_envs, t)
                    self.writer.add_scalar(
                        "success",
                        is_success_since_last_log / log_interval,
                        t * self.n_envs,
                    )
                    reward_since_last_log = 0.0
                    is_success_since_last_log = 0.0
                if done.any():
                    self.writer.add_scalar(
                        "episode_length", episode_lengths[done].mean(), t * self.n_envs
                    )
                    self.writer.add_scalar(
                        "return", episode_return[done].mean(), t * self.n_envs
                    )
                    episode_lengths[done] = 0
                    episode_return[done] = 0.0

            cum_reward += reward

            if record_transitions:
                # note that the final observation of an episode is not included in obs_seq
                obs_seq[t + 1] = obs_next
                act_seq[t] = action
                rew_seq[t] = reward
                term_seq[t] = terminated
                trunc_seq[t] = truncated

            if "_final_observation" in self.info:
                # provide final observation of episode to hierarchy
                obs_next_transition = copy_torch_or_numpy(obs_next)
                for i, env_final_obs in enumerate(self.info["final_observation"]):
                    if env_final_obs is not None:
                        obs_next_transition[i] = env_final_obs
            else:
                obs_next_transition = obs_next

            # process transition in hierarchy, i.e. add to replay buffer etc.
            self.hierarchy.process_transition(
                obs_next=obs_next_transition,
                rew=reward,
                terminated=terminated,
                truncated=truncated,
            )

            if learn:
                self.hierarchy.learn()

            self.obs = obs_next
            n_episodes[done] += 1
            self.save_checkpoint(t, n_steps)

            if test_interval is not None and t % test_interval == 0:
                # test
                test_hierarchy = self.hierarchy.get_copy()
                self._single_test_stats = self.test(
                    t,
                    test_hierarchy,
                    n_test_steps,
                    env_expects_numpy=env_expects_numpy,
                    video_logger=video_logger,
                )
                for key, value in self._single_test_stats.items():
                    test_stats[key].append(value)

            if verbose:
                postfix_dict = {
                    "mean_return": (cum_reward / np.maximum(n_episodes, 1)).mean(),
                    "mean_episode_length": ((t + 1) / np.maximum(n_episodes, 1)).mean(),
                }
                if self._single_test_stats is not None:
                    postfix_dict["test_return"] = self._single_test_stats["test_return"]
                    if self.success_key is not None:
                        postfix_dict["test_success"] = self._single_test_stats[
                            "test_success_in_ep"
                        ]
                step_range.set_postfix(postfix_dict)

        n_last_tests = min(len(test_stats["test_reward"]), 5)
        stats = {
            "n_episodes": n_episodes.sum(),
            "mean_reward": (cum_reward / n_steps).mean(),
            "mean_episode_length": (n_steps / n_episodes.clip(1)).mean(),
            "mean_return": (cum_reward / n_episodes.clip(1)).mean(),
        }
        if n_last_tests > 0:
            stats.update(
                {
                    "mean_test_return": np.mean(test_stats["test_return"]),
                    "mean_test_reward": np.mean(test_stats["test_reward"]),
                    "last_test_return": test_stats["test_return"][-1],
                    "last_test_reward": test_stats["test_reward"][-1],
                    "last_5_test_return": np.mean(
                        test_stats["test_return"][-n_last_tests:]
                    ),
                    "last_5_test_reward": np.mean(
                        test_stats["test_reward"][-n_last_tests:]
                    ),
                }
            )

        if record_transitions:
            batch = Batch(
                obs=obs_seq,
                act=act_seq,
                rew=rew_seq,
                terminated=term_seq,
                truncated=trunc_seq,
            )
            return stats, batch
        else:
            return stats
