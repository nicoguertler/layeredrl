import gymnasium as gym
import torch

from layeredrl.hierarchies import RandomHierarchy, FlatTianshouHierarchy


def rollout(env, hierarchy):
    obs, info = env.reset()
    obs = torch.tensor(obs).unsqueeze(0)
    done = False
    n_steps = 0
    while not done:
        with torch.no_grad():
            action = hierarchy.get_action(obs)
        action = action.cpu().numpy()
        obs_next, reward, terminated, truncated, info = env.step(action[0])
        if terminated or truncated:
            done = True
        obs_next = torch.tensor(obs_next).unsqueeze(0)
        reward = torch.tensor([reward])
        terminated = torch.tensor([terminated], dtype=torch.bool)
        truncated = torch.tensor([truncated], dtype=torch.bool)
        hierarchy.process_transition(obs_next, reward, terminated, truncated)
        obs = obs_next
        n_steps += 1
    return n_steps


def test_random_hierarchy():
    torch.manual_seed(0)
    # discrete mountain car
    env = gym.make("MountainCar-v0")
    env.action_space.seed(123)
    random_hierarchy = RandomHierarchy(env, device=torch.device("cpu"))
    random_hierarchy.reset()
    n_steps = rollout(env, random_hierarchy)
    assert n_steps <= 200

    # continuous mountain car
    env = gym.make("MountainCarContinuous-v0")
    env.action_space.seed(123)
    random_hierarchy = RandomHierarchy(env, device=torch.device("cpu"))
    random_hierarchy.reset()
    n_steps = rollout(env, random_hierarchy)
    assert n_steps <= 999


def test_flat_tianshou_hierarchy():
    torch.manual_seed(0)
    # discrete mountain car
    env = gym.make("MountainCar-v0")
    env.action_space.seed(123)
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
            "dueling_param": (
                {"hidden_sizes": [128, 128]},
                {"hidden_sizes": [128, 128]},
            ),
        },
        "optims": {
            "critic_optim": {
                "_target_": "torch.optim.Adam",
                "lr": 0.013,
            }
        },
        "policy": {
            "_target_": "tianshou.policy.DQNPolicy",
            "_partial_": True,
            "discount_factor": 0.99,
            "estimation_step": 1,
            "target_update_freq": 500,
        },
        "policy_dynamic_args": {
            "model": "__critic_0__",
            "optim": "__critic_optim_0__",
        },
    }
    flat_tianshou_hierarchy = FlatTianshouHierarchy(
        env, tianshou_config, device=torch.device("cpu"), buffer_size=1000
    )
    flat_tianshou_hierarchy.reset()
    n_steps = rollout(env, flat_tianshou_hierarchy)
    assert n_steps <= 200
    # make sure learning actually changes the parameters
    params_before_update = []
    for param in flat_tianshou_hierarchy.levels[0].critics[0].parameters():
        params_before_update.append(param.clone())
    flat_tianshou_hierarchy.learn()
    for param_before, param_after in zip(
        params_before_update, flat_tianshou_hierarchy.levels[0].critics[0].parameters()
    ):
        assert not torch.all(torch.eq(param_before, param_after))

    # continuous mountain car
    env = gym.make("MountainCarContinuous-v0")
    env.action_space.seed(123)
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
        },
        "optims": {
            "actor_optim": {
                "_target_": "torch.optim.Adam",
            },
            "critic_optim": {
                "_target_": "torch.optim.Adam",
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
    flat_tianshou_hierarchy = FlatTianshouHierarchy(
        env, tianshou_config, device=torch.device("cpu"), buffer_size=1000
    )
    flat_tianshou_hierarchy.reset()
    n_steps = rollout(env, flat_tianshou_hierarchy)
    assert n_steps <= 999
    # make sure learning actually changes the parameters
    params_before_update = []
    for param in flat_tianshou_hierarchy.levels[0].actor.parameters():
        params_before_update.append(param.clone())
    flat_tianshou_hierarchy.learn()
    for param_before, param_after in zip(
        params_before_update, flat_tianshou_hierarchy.levels[0].actor.parameters()
    ):
        assert not torch.all(torch.eq(param_before, param_after))


if __name__ == "__main__":
    test_random_hierarchy()
    test_flat_tianshou_hierarchy()
