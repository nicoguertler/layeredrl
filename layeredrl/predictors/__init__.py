from functools import partial

import gymnasium as gym

from .predictor import Predictor
from .predictor_factory import PredictorFactory
from .reward_predictor import RewardPredictor
from .static_predictor import StaticPredictor
from ..nets import ProbFCDynamics, RewardNet, FixedEncoderNet, ValueNet
from ..models import ProbabilisticEnsemble
from ..utils.misc import get_obs_indices


def get_default_predictor_factory(
    env: gym.Env, sb_start_duration: float
) -> PredictorFactory:
    """Get a default predictor factory.

    The predictor factory creates a RewardPredictor object. The method assumes that
    the environment is goal-based and interprets the desired goal as the context
    and the achieved goal as the state for the planner level.

    Args:
        env: The environment for which to create the predictor factory.
        sb_start_duration: Duration (in env steps) for symmetry breaking start.
    Returns:
        The predictor factory.
    """
    # Get index ranges in flattened observation for keys of observation dict
    obs_indices, _ = get_obs_indices(env)

    # Networks for the predictor (partial because correct dimensions are automatically set
    # during assembly of the hierarchy)
    partial_net = partial(ProbFCDynamics)
    partial_model = partial(
        ProbabilisticEnsemble,
        partial_net=partial_net,
        symmetry_breaking_start=True,
        sb_start_duration=sb_start_duration,
        sb_start_factor=1.0,
        n_models=1,
        n_modes=4,
        n_particles_per_model=1,
        normalize_targets=True,
        target_bn_momentum=0.001,
    )
    partial_val_func = partial(ValueNet)
    partial_rew_func = partial(RewardNet)
    # Pick out desired goal as context and position and velocity as latent state
    partial_encoder = partial(
        FixedEncoderNet,
        latent_state_dims=range(*obs_indices["achieved_goal"]),
        context_dims=range(*obs_indices["desired_goal"]),
    )

    # Predictor factory
    partial_predictor = partial(
        RewardPredictor,
        learn_encoder=False,
    )
    predictor_factory = PredictorFactory(
        partial_model=partial_model,
        partial_val_func=partial_val_func,
        partial_rew_func=partial_rew_func,
        partial_encoder=partial_encoder,
        partial_predictor=partial_predictor,
    )
    return predictor_factory


__all__ = [
    "Predictor",
    "PredictorFactory",
    "RewardPredictor",
    "StaticPredictor",
    "get_default_predictor_factory",
]
