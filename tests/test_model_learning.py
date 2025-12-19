import torch

from gymnasium.spaces import Box, Discrete
from tianshou.data import Batch

from layeredrl.models.probabilistic_ensemble import ProbabilisticEnsemble
from layeredrl.nets import ProbFCDynamics


# discrete actions
action_offset = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]) / 100.0


def learn_model(
    n_models,
    n_particles_per_model,
    batch_size=128,
    device=torch.device("cpu"),
    discrete_actions=False,
):

    # create model
    def net_factory(state_space, context_space, action_space, n_modes):
        return ProbFCDynamics(
            state_space,
            context_space,
            action_space,
            n_modes=n_modes,
            hidden_sizes=[32, 32],
            device=device,
        )

    action_space = Discrete(4) if discrete_actions else Box(-1, 1, (2,))
    model = ProbabilisticEnsemble(
        state_space=Box(-1, 1, (2,)),
        context_space=Box(-1, 1, (1,)),
        action_space=action_space,
        n_models=n_models,
        n_particles_per_model=n_particles_per_model,
        partial_net=net_factory,
        learning_rate=1e-4,
        create_optimizer=True,
        normalize_targets=False,
        device=device,
    )

    # train model
    for _ in range(1000):
        # generate data
        states = torch.rand(batch_size, 2) * 2 - 1
        contexts = torch.sign(torch.rand(batch_size, 1) * 2 - 1)
        if discrete_actions:
            actions = torch.randint(0, 4, (batch_size, 1))
            states_next = states + action_offset[actions.squeeze()]
            rewards = (actions[:, 0] == 0) * contexts.squeeze()
        else:
            actions = torch.rand(batch_size, 2) * 2 - 1
            states_next = states + actions / 100.0
            rewards = actions[:, 0] * contexts.squeeze()
        batch = Batch(
            state=states,
            context=contexts,
            act=actions,
            state_next=states_next,
            rew=rewards,
            terminated=torch.zeros(batch_size),
        )
        model.learn([batch])

    # check model error
    test_batch_size = 200
    states = torch.rand(test_batch_size, 2) * 2 - 1
    contexts = torch.sign(torch.rand(test_batch_size, 1) * 2 - 1)
    if discrete_actions:
        actions = torch.randint(0, 4, (test_batch_size, 1))
        states_next = states + action_offset[actions.squeeze()]
        rewards = (actions[:, 0] == 0) * contexts.squeeze()
    else:
        actions = torch.rand(test_batch_size, 2) * 2 - 1
        states_next = states + actions / 100.0
        rewards = actions[:, 0] * contexts.squeeze()
    with torch.no_grad():
        pred_means, weights, _, _ = model.predict(states, contexts, actions)
        # multiply with weights of modes and average over particles
        states_pred = (pred_means * weights[..., None]).sum(dim=-2).mean(dim=1)
        state_error = torch.norm(states_next - states_pred, dim=-1).mean()
    print("state prediction error:", state_error.item())
    assert state_error < 1e-2


def test_learn_probabilistic_ensemble():
    n_models = 2
    n_particles_per_model = 2
    learn_model(
        n_models=n_models,
        n_particles_per_model=n_particles_per_model,
        discrete_actions=False,
    )
    learn_model(
        n_models=n_models,
        n_particles_per_model=n_particles_per_model,
        discrete_actions=True,
    )


if __name__ == "__main__":
    test_learn_probabilistic_ensemble()
