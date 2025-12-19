from functools import partial

from gymnasium.spaces import Box, Discrete
import torch

from layeredrl.levels import RandomLevel, TianshouLevel, SPlaTESLevel, DADSLevel
from layeredrl.models import ProbabilisticEnsemble
from layeredrl.nets import ProbFCDynamics, IdentityEncoder
from layeredrl.predictors import StaticPredictor


def test_random_level():
    state_act_dim = 2
    action_space = Box(low=-1, high=1, shape=(state_act_dim,))
    observation_space = Box(low=-1, high=1, shape=(state_act_dim,))
    random_level = RandomLevel()
    random_level.initialize(
        action_space=action_space,
        env_obs_space=observation_space,
        n_env_instances=1,
        parent_predictor=None,
    )
    random_level.reset()
    assert random_level.action_space == action_space
    for _ in range(100):
        action = random_level.get_action(
            torch.Tensor(action_space.sample()).unsqueeze(0), None, None
        )[0]
        action = action.squeeze(0).cpu().numpy()
        assert action_space.contains(action)


tianshou_config_continous = {
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
        "_target_": "tianshou.utils.net.continuous.Critic",
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


def test_tianshou_level_continuous():
    """Test Tianshou level with continuous action space"""
    state_act_dim = 2
    observation_space = Box(low=-1.0, high=1.0, shape=(state_act_dim,))
    action_space = Box(low=-2.0, high=2.0, shape=(state_act_dim,))
    tianshou_level = TianshouLevel(tianshou_config=tianshou_config_continous)
    tianshou_level.initialize(
        action_space=action_space,
        env_obs_space=observation_space,
        n_env_instances=1,
        parent_predictor=None,
    )
    tianshou_level.reset()
    assert tianshou_level.action_space == action_space
    with torch.no_grad():
        for _ in range(100):
            obs = torch.tensor(observation_space.sample()).unsqueeze(0)
            action = tianshou_level.get_action(obs, None, None)[0]
            assert action_space.contains(action.squeeze().cpu().numpy())


tianshou_config_discrete = {
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


def test_tianshou_level_discrete():
    """Test Tianshou level with discrete action space"""
    action_space = Discrete(n=5)
    observation_space = Box(low=-1.0, high=1.0, shape=(2,))
    tianshou_level = TianshouLevel(tianshou_config=tianshou_config_discrete)
    tianshou_level.initialize(
        action_space=action_space,
        env_obs_space=observation_space,
        n_env_instances=1,
        parent_predictor=None,
    )
    tianshou_level.reset()
    assert tianshou_level.action_space == action_space
    with torch.no_grad():
        for _ in range(100):
            obs = torch.Tensor(observation_space.sample()).unsqueeze(0)
            action = tianshou_level.get_action(obs, None, None)[0]
            assert action_space.contains(action.squeeze().cpu().numpy())


tdmpc2_config = {
    "obs": "state",
    # training
    "batch_size": 256,
    "reward_coef": 0.1,
    "value_coef": 0.1,
    "consistency_coef": 20,
    "term_coef": 0.05,
    "rho": 0.5,
    "lr": 3.0e-4,
    "enc_lr_scale": 0.3,
    "grad_clip_norm": 20,
    "tau": 0.01,
    "discount_denom": 5,
    "discount_min": 0.95,
    "discount_max": 0.995,
    "discount_factor": [0.99],
    "buffer_size": 1000000,
    "prioritized_sampling": False,
    "alpha": 1.0,
    "beta": 0.5,
    "inter_skill_bootstrap": True,
    # planning
    "mpc": True,
    "iterations": 6,
    "num_samples": 512,
    "num_elites": 64,
    "num_pi_trajs": 24,
    "horizon": 3,
    "min_std": 0.05,
    "max_std": 2,
    "temperature": 0.5,
    # actor
    "log_std_min": -10,
    "log_std_max": 2,
    "entropy_coef": 1.0e-4,
    # critic
    "num_bins": 101,
    "vmin": -11,
    "vmax": +11,
    # architecture
    "model_size": "???",
    "num_enc_layers": 2,
    "enc_dim": 256,
    "num_channels": 32,
    "mlp_dim": 512,
    "latent_dim": 512,
    "task_dim": 96,
    "num_q": 5,
    "dropout": 0.01,
    "simnorm_dim": 8,
    # convenience
    "multitask": False,
}


def test_dads_level_continuous():
    state_act_dim = 2
    skill_space_dim = 2
    skill_space = Box(low=-1.0, high=1.0, shape=(skill_space_dim,))
    observation_space = Box(low=-1.0, high=1.0, shape=(state_act_dim,))
    context_space = Box(low=-1.0, high=1.0, shape=(1,))
    action_space = Box(low=-2.0, high=2.0, shape=(state_act_dim,))
    model = ProbabilisticEnsemble(
        state_space=observation_space,
        context_space=context_space,
        action_space=skill_space,
        partial_net=partial(
            ProbFCDynamics,
        ),
    )
    encoder = IdentityEncoder(observation_space.shape)
    predictor = StaticPredictor(
        model=model,
        val_func=None,
        rew_func=None,
        encoder=encoder,
        latent_state_dim=encoder.latent_state_dim,
        context_dim=encoder.context_dim,
    )
    dads_level = DADSLevel(
        skill_space_dim=skill_space_dim,
        control_interval=10,
        tdmpc2_config=tdmpc2_config,
    )
    dads_level.initialize(
        action_space=action_space,
        env_obs_space=observation_space,
        n_env_instances=1,
        parent_predictor=predictor,
    )
    dads_level.reset()
    assert dads_level.action_space == action_space
    with torch.no_grad():
        for _ in range(100):
            obs = torch.tensor(observation_space.sample()).unsqueeze(0)
            z = torch.rand(1, skill_space_dim)
            action = dads_level.get_action(obs, z, None)[0]
            assert action_space.contains(action.squeeze().cpu().numpy())


def test_splates_level_continuous():
    state_act_dim = 2
    skill_space_dim = 2
    skill_space = Box(low=-1.0, high=1.0, shape=(skill_space_dim,))
    observation_space = Box(low=-1.0, high=1.0, shape=(state_act_dim,))
    context_space = Box(low=-1.0, high=1.0, shape=(1,))
    action_space = Box(low=-2.0, high=2.0, shape=(state_act_dim,))
    model = ProbabilisticEnsemble(
        state_space=observation_space,
        context_space=context_space,
        action_space=skill_space,
        partial_net=partial(
            ProbFCDynamics,
        ),
    )
    encoder = IdentityEncoder(observation_space.shape)
    predictor = StaticPredictor(
        model=model,
        val_func=None,
        rew_func=None,
        encoder=encoder,
        latent_state_dim=2,
        context_dim=2,
    )
    splates_level = SPlaTESLevel(
        skill_space_dim=skill_space_dim,
        control_interval=10,
        tdmpc2_config=tdmpc2_config,
    )
    splates_level.initialize(
        action_space=action_space,
        env_obs_space=observation_space,
        n_env_instances=1,
        parent_predictor=predictor,
    )
    splates_level.reset()
    assert splates_level.action_space == action_space
    with torch.no_grad():
        for _ in range(100):
            obs = torch.tensor(observation_space.sample()).unsqueeze(0)
            z = torch.rand(1, skill_space_dim)
            action = splates_level.get_action(obs, z, None)[0]
            assert action_space.contains(action.squeeze().cpu().numpy())


if __name__ == "__main__":
    test_random_level()
    test_tianshou_level_continuous()
    test_tianshou_level_discrete()
    test_splates_level_continuous()
    test_dads_level_continuous()
