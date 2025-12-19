import numpy as np
from tianshou.data import Batch, ReplayBuffer
import torch
from torch import nn

from gymnasium.spaces import Box

from layeredrl.models import ProbabilisticEnsemble
from layeredrl.nets import ProbFCDynamics, EncoderNet
from layeredrl.predictors import Predictor, RewardPredictor
from layeredrl.utils.misc import to_torch


def test_predictor():
    np.random.seed(0)
    buffer = ReplayBuffer(10)
    for i in range(12):
        buffer.add(
            Batch(obs=0, act=0, rew=i + 1, terminated=i % 4 == 3, truncated=False)
        )
    indices, traj_len, validity_mask = Predictor.sample_partial_trajectories(
        buffer=buffer, batch_size=2, n_steps=3
    )


def get_n_step_returns(buffer, indices, k_steps, n_steps, gamma, q_func, repr_map):
    batch_size = indices.shape[1]
    unfinished_index = buffer.unfinished_index()
    returns = torch.zeros((batch_size, k_steps))
    in_episode = torch.ones((batch_size,), dtype=float)
    for i in range(k_steps):
        terminated = torch.zeros((batch_size,), dtype=float)
        truncated = torch.zeros((batch_size,), dtype=float)
        valid_steps = torch.zeros((batch_size,), dtype=float)
        for j in range(n_steps):
            valid_steps += 1.0 - truncated
            current_indices = indices[i + j, :]
            returns[:, i] += (
                to_torch(buffer.rew[current_indices])
                * gamma**j
                * (1.0 - terminated)
                * (1.0 - truncated)
            )
            terminated[buffer.terminated[current_indices]] = 1.0
            truncated[buffer.truncated[current_indices]] = 1.0
            # treat unfinished index as end of truncated episode
            truncated[current_indices == unfinished_index] = 1.0
        bootstrap_indices = indices[i + n_steps, :]
        # bootstrapping
        bootstrap_gamma = gamma**valid_steps
        returns[:, i] += (
            bootstrap_gamma
            * q_func(
                *repr_map(to_torch(buffer.obs[bootstrap_indices])),
                to_torch(buffer.act[bootstrap_indices])
            )
            * (1.0 - terminated)
        )
        returns[:, i] *= in_episode
        # is the index corresponding to i the end of the episode?
        i_indices = indices[i, :]
        end_of_episode = np.logical_or.reduce(
            (
                buffer.done[i_indices],
                i_indices == unfinished_index,
                buffer.truncated[i_indices],
            )
        )
        in_episode = torch.minimum(in_episode, 1.0 - to_torch(end_of_episode).float())
    return returns


class DummyQFunc(nn.Module):
    def forward(self, s, c, a):
        return s.squeeze()


class DummyRewardFunc(nn.Module):
    def forward(self, s, c, a):
        return torch.zeros(s.shape[0])


def net_factory(s_space, c_space, a_space, n_modes):
    return ProbFCDynamics(
        state_space=s_space,
        context_space=c_space,
        action_space=a_space,
        n_modes=n_modes,
        hidden_sizes=[48, 48],
    )


def test_reward_predictor_termination():
    np.random.seed(0)
    buffer = ReplayBuffer(12)
    for i in range(14):
        buffer.add(
            Batch(
                obs=[i],
                act=[0],
                rew=i + 1,
                terminated=i % 6 == 3,
                truncated=False,
                info={"random_mode": False},
            )
        )

    q_func = DummyQFunc()
    rew_func = DummyRewardFunc()

    def encoder(obs):
        return obs, torch.zeros_like(obs)

    state_dim = 1
    context_dim = 1
    action_dim = 1
    model = ProbabilisticEnsemble(
        state_space=Box(-1.0, 1.0, (state_dim,)),
        context_space=Box(-1.0, 1.0, (context_dim,)),
        action_space=Box(-1.0, 1.0, (action_dim,)),
        partial_net=net_factory,
    )
    predictor = RewardPredictor(
        model=model,
        val_func=q_func,
        rew_func=rew_func,
        encoder=encoder,
        latent_state_dim=1,
        context_dim=1,
        learn_encoder=False,
    )
    batch_size = 4
    k_steps = 3
    n_steps = 2
    gamma = 0.99
    targets = predictor.compute_targets(
        buffer=buffer,
        batch_size=batch_size,
        k_steps=k_steps,
        n_steps=n_steps,
        gamma=gamma,
    )
    assert targets.rew.shape == (batch_size, k_steps)
    assert targets.indices.shape == (k_steps + n_steps, batch_size)
    assert targets.traj_len.shape == (batch_size,)

    returns = get_n_step_returns(
        buffer, targets.indices, k_steps, n_steps, gamma, q_func, encoder
    )

    assert torch.allclose(targets.value, returns)
    for i in range(k_steps):
        assert torch.all(targets.rew[:, i] == to_torch(buffer.rew[targets.indices[i]]))


def test_reward_predictor_truncation():
    np.random.seed(0)
    buffer = ReplayBuffer(12)
    for i in range(14):
        buffer.add(
            Batch(
                obs=[i],
                act=[0],
                rew=i + 1,
                terminated=False,
                truncated=i % 6 == 3,
                info={"random_mode": False},
            )
        )

    q_func = DummyQFunc()
    rew_func = DummyRewardFunc()

    def encoder(obs):
        return obs, torch.zeros_like(obs)

    state_dim = 1
    context_dim = 1
    action_dim = 1
    model = ProbabilisticEnsemble(
        state_space=Box(-1.0, 1.0, (state_dim,)),
        context_space=Box(-1.0, 1.0, (context_dim,)),
        action_space=Box(-1.0, 1.0, (action_dim,)),
        partial_net=net_factory,
    )
    predictor = RewardPredictor(
        model=model,
        val_func=q_func,
        rew_func=rew_func,
        encoder=encoder,
        latent_state_dim=1,
        context_dim=1,
        learn_encoder=False,
    )
    batch_size = 4
    k_steps = 3
    n_steps = 2
    gamma = 0.99
    targets = predictor.compute_targets(
        buffer=buffer,
        batch_size=batch_size,
        k_steps=k_steps,
        n_steps=n_steps,
        gamma=gamma,
    )
    assert targets.rew.shape == (batch_size, k_steps)
    assert targets.indices.shape == (k_steps + n_steps, batch_size)
    assert targets.traj_len.shape == (batch_size,)

    returns = get_n_step_returns(
        buffer, targets.indices, k_steps, n_steps, gamma, q_func, encoder
    )

    assert torch.allclose(targets.value, returns)
    for i in range(k_steps):
        assert torch.all(targets.rew[:, i] == to_torch(buffer.rew[targets.indices[i]]))


class QFunc(nn.Module):
    def __init__(self, state_dim, context_dim, action_dim, hidden_sizes):
        super().__init__()
        layers = []
        input_dim = state_dim + context_dim + action_dim
        for i in range(len(hidden_sizes)):
            layers.append(nn.Linear(input_dim, hidden_sizes[i]))
            layers.append(nn.ReLU())
            input_dim = hidden_sizes[i]
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, s, c, a):
        return self.mlp(torch.cat([s, c, a], dim=-1)).squeeze()


def test_reward_predictor_learning():
    obs_dim = 3
    state_dim = 2
    context_dim = 1
    action_dim = 2
    buffer_size = 10000

    def init_obs():
        obs = np.random.rand(obs_dim).astype(np.float32)
        obs[-1] -= 0.5
        return obs

    np.random.seed(0)
    torch.manual_seed(0)
    buffer = ReplayBuffer(buffer_size)
    obs = init_obs()
    obs_next = obs.copy()
    for i in range(buffer_size):
        act = np.random.rand(action_dim).astype(np.float32)
        terminated = obs[0] > 1.0
        obs_next[:state_dim] += 0.05 * act
        rew = np.sign(obs[-1])
        buffer.add(
            Batch(
                obs=obs,
                obs_next=obs_next,
                act=act,
                rew=rew,
                terminated=terminated,
                truncated=False,
                info={"random_mode": False},
            )
        )
        obs = obs_next
        if terminated:
            obs = init_obs()
            obs_next = obs.copy()

    q_func = QFunc(state_dim, context_dim, action_dim, [48, 48])
    rew_func = QFunc(state_dim, context_dim, action_dim, [48, 48])
    encoder = EncoderNet((obs_dim,), state_dim, context_dim)

    model = ProbabilisticEnsemble(
        state_space=Box(-1.0, 1.0, (state_dim,)),
        context_space=Box(-1.0, 1.0, (context_dim,)),
        action_space=Box(-1.0, 1.0, (action_dim,)),
        partial_net=net_factory,
    )
    batch_size = 64
    k_steps = 3
    n_steps = 2
    gamma = 0.90
    tau = 0.005
    predictor = RewardPredictor(
        model=model,
        val_func=q_func,
        rew_func=rew_func,
        encoder=encoder,
        latent_state_dim=state_dim,
        context_dim=context_dim,
        k_steps=k_steps,
        n_steps=n_steps,
        gamma=gamma,
        tau=tau,
        reward_lr=1e-3,
        value_lr=1e-3,
        encoder_lr=1e-3,
        consistency_loss_weight=0.01,
        value_loss_weight=0.01,
    )
    targets = predictor.compute_targets(
        buffer=buffer,
        batch_size=batch_size,
        k_steps=k_steps,
        n_steps=n_steps,
        gamma=gamma,
    )
    assert targets.rew.shape == (batch_size, k_steps)
    assert targets.indices.shape == (k_steps + n_steps, batch_size)
    assert targets.traj_len.shape == (batch_size,)

    losses = 0
    for i in range(500):
        loss, _ = predictor.learn(
            buffer=buffer,
            n_steps=n_steps,
            batch_size=batch_size,
            model_batch_size=batch_size,
            n_total_env_steps=i,
        )
        losses += loss
        if i % 100 == 0:
            print(losses / 100)
            losses = 0

    # test reward and value prediction
    batch, _ = buffer.sample(1000)
    with torch.no_grad():
        _, _, _, rewards, _ = predictor.predict(
            to_torch(batch.obs), to_torch(batch.act)
        )
        values = predictor.value(to_torch(batch.obs), to_torch(batch.act))
        sign_corr_rew = np.sign(
            rewards.detach().cpu().numpy() * batch.obs[:, -1]
        ).mean()
        sign_corr_val = np.sign(values.detach().cpu().numpy() * batch.obs[:, -1]).mean()
        print("reward sign correlation: ", sign_corr_rew)
        print("value sign correlation: ", sign_corr_val)
    assert sign_corr_rew > 0.9
    assert sign_corr_val > 0.9


if __name__ == "__main__":
    test_predictor()
    test_reward_predictor_termination()
    test_reward_predictor_truncation()
    test_reward_predictor_learning()
