import gymnasium as gym
import torch

from layeredrl.hierarchies import RandomHierarchy
from layeredrl.collectors import Collector


def test_collector_discrete_actions_single_env():
    torch.manual_seed(0)
    # discrete mountain car
    env = gym.make("MountainCar-v0")
    env.action_space.seed(123)
    random_hierarchy = RandomHierarchy(env, device=torch.device("cpu"))
    collector = Collector(
        hierarchy=random_hierarchy, env=env, device=torch.device("cpu")
    )
    collector.reset()
    n_steps = 2000
    stats, batch = collector.collect(n_steps=n_steps, record_transitions=True)

    # + 1 for initial observation
    assert batch.obs.shape[0] == n_steps + 1
    assert batch.act.shape[0] == n_steps
    assert batch.terminated.sum() == 0
    assert batch.truncated.sum() == 1 * n_steps // 200
    assert batch.truncated.sum() == stats["n_episodes"]


def test_collector_discrete_actions():
    torch.manual_seed(0)
    # discrete mountain car
    n_envs = 3
    env = gym.make_vec(
        id="MountainCar-v0",
        num_envs=n_envs,
        vectorization_mode="async",
        render_mode=None,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )
    env.action_space.seed(123)
    random_hierarchy = RandomHierarchy(env, device=torch.device("cpu"))
    collector = Collector(
        hierarchy=random_hierarchy, env=env, device=torch.device("cpu")
    )
    collector.reset()
    n_steps = 2000
    stats, batch = collector.collect(n_steps=n_steps, record_transitions=True)

    # + 1 for initial observation
    assert batch.obs.shape[0] == n_steps + 1
    assert batch.act.shape[0] == n_steps
    assert batch.terminated.sum() == 0
    assert batch.truncated.sum() == n_envs * n_steps // 200
    assert batch.truncated.sum() == stats["n_episodes"]


def test_collector_continuous_actions():
    torch.manual_seed(0)
    # continuous mountain car
    n_envs = 3
    env = gym.make_vec(
        id="MountainCarContinuous-v0",
        num_envs=n_envs,
        vectorization_mode="async",
        render_mode=None,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
    )
    env.action_space.seed(123)
    random_hierarchy = RandomHierarchy(env, device=torch.device("cpu"))
    collector = Collector(
        hierarchy=random_hierarchy, env=env, device=torch.device("cpu")
    )
    collector.reset()
    n_steps = 2000
    stats, batch = collector.collect(n_steps=n_steps, record_transitions=True)

    # + 1 for initial observation
    assert batch.obs.shape[0] == n_steps + 1
    assert batch.act.shape[0] == n_steps
    assert batch.terminated.sum() == 0
    assert batch.truncated.sum() == n_envs * (n_steps // 999)
    assert batch.truncated.sum() == stats["n_episodes"]


if __name__ == "__main__":
    test_collector_discrete_actions_single_env()
    test_collector_discrete_actions()
    test_collector_continuous_actions()
