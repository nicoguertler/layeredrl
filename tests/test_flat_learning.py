import gymnasium as gym
import numpy as np
import torch

from layeredrl.hierarchies import FlatTianshouHierarchy
from layeredrl.collectors import Collector


def test_flat_learning_contiuous(device="cpu"):
    torch.manual_seed(0)
    np.random.seed(0)

    n_envs = 3
    env = gym.make_vec(
        id="InvertedPendulum-v5",
        num_envs=n_envs,
        vectorization_mode="async",
        render_mode=None,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )
    env.reset(seed=42)
    tianshou_config = {
        "n_critics": 2,
        "actor": {
            "_target_": "tianshou.utils.net.continuous.ActorProb",
            "preprocess_net": {
                "_target_": "layeredrl.nets.ConcatNet",
                "mapped_env_obs_shape": "__mapped_env_obs_shape__",
                "level_input_dim": "__level_input_dim__",
                "level_state_dims": "__level_state_dims__",
                "hidden_sizes": [32, 32],
                "device": "__device__",
            },
            "action_shape": "__action_dim__",
            "device": "__device__",
            "unbounded": True,
            "conditioned_sigma": True,
        },
        "critic": {
            "_target_": "layeredrl.nets.Critic",
            "preprocess_net": {
                "_target_": "layeredrl.nets.ConcatNet",
                "mapped_env_obs_shape": "__mapped_env_obs_shape__",
                "level_input_dim": "__level_input_dim__",
                "level_state_dims": "__level_state_dims__",
                "action_shape": "__action_dim__",
                "hidden_sizes": [32, 32],
                "concat": True,
                "device": "__device__",
            },
            "device": "__device__",
        },
        "optims": {
            "actor_optim": {
                "_target_": "torch.optim.Adam",
                "lr": 0.003,
            },
            "critic_optim": {
                "_target_": "torch.optim.Adam",
                "lr": 0.003,
            },
        },
        "policy": {
            "_target_": "tianshou.policy.SACPolicy",
            "_partial_": True,
            "tau": 0.005,
            "gamma": 0.99,
            "alpha": 0.2,
            "estimation_step": 1,
        },
        "policy_dynamic_args": {
            "actor": "__actor__",
            "actor_optim": "__actor_optim__",
            "critic1": "__critic_0__",
            "critic1_optim": "__critic_optim_0__",
            "critic2": "__critic_1__",
            "critic2_optim": "__critic_optim_1__",
            "action_space": "__action_space__",
        },
    }
    hierarchy = FlatTianshouHierarchy(
        env, tianshou_config, device=torch.device(device), buffer_size=5000
    )
    collector = Collector(hierarchy=hierarchy, env=env, device=torch.device(device))
    collector.reset()
    stats = collector.collect(n_steps=5000, learn=True, record_transitions=False)
    print("training stats: \n", stats)
    stats_final = collector.collect(n_steps=1000, learn=False, record_transitions=False)
    print("final stats: \n", stats_final)
    assert stats_final["mean_return"] > 600


def test_flat_learning_discrete(device="cpu"):
    torch.manual_seed(0)
    np.random.seed(0)

    n_envs = 3
    env = gym.make_vec(
        id="CartPole-v1",
        num_envs=n_envs,
        vectorization_mode="async",
        render_mode=None,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )
    env.reset(seed=42)
    tianshou_config = {
        "n_critics": 1,
        "critic": {
            "_target_": "layeredrl.nets.ConcatNet",
            "mapped_env_obs_shape": "__mapped_env_obs_shape__",
            "level_input_dim": "__level_input_dim__",
            "level_state_dims": "__level_state_dims__",
            "action_shape": "__action_dim__",
            "hidden_sizes": [32, 32],
            "device": "__device__",
        },
        "optims": {
            "critic_optim": {
                "_target_": "torch.optim.Adam",
                "lr": 0.001,
            }
        },
        "policy": {
            "_target_": "tianshou.policy.DQNPolicy",
            "_partial_": True,
            "discount_factor": 0.99,
            "estimation_step": 1,
            "target_update_freq": 100,
        },
        "policy_dynamic_args": {
            "model": "__critic_0__",
            "optim": "__critic_optim_0__",
        },
        "eps": 0.3,
    }
    hierarchy = FlatTianshouHierarchy(
        env, tianshou_config, device=torch.device(device), buffer_size=5000
    )
    collector = Collector(hierarchy=hierarchy, env=env, device=torch.device(device))
    collector.reset()
    stats = collector.collect(n_steps=5000, learn=True, record_transitions=False)
    print("training stats: \n", stats)
    stats_final = collector.collect(n_steps=1000, learn=False, record_transitions=False)
    print("final stats: \n", stats_final)
    assert stats_final["mean_return"] > 200


if __name__ == "__main__":
    test_flat_learning_contiuous()
    test_flat_learning_discrete()
