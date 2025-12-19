from contextlib import contextmanager
from copy import deepcopy
from typing import Tuple, Dict, Union

import gymnasium as gym
import numpy as np
import torch


def to_torch(
    array: Union[np.ndarray, torch.Tensor], device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Convert an array to a torch tensor.

    Args:
        array: The array.
    Returns:
        The torch tensor.
    """
    if isinstance(array, torch.Tensor):
        return array
    else:
        return torch.as_tensor(array, device=device)


def to_numpy(
    array: Union[np.ndarray, torch.Tensor], device: torch.device = torch.device("cpu")
) -> np.ndarray:
    """Convert an array to a numpy array.

    Args:
        array: The array.
    Returns:
        The numpy array.
    """
    if isinstance(array, np.ndarray):
        return array
    else:
        return array.cpu().numpy()


def copy_torch_or_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Copy a torch tensor or numpy array.

    Args:
        array: The array.
    Returns:
        The copy.
    """
    if isinstance(array, np.ndarray):
        return array.copy()
    else:
        return array.clone()


@contextmanager
def temp_eval_mode(net):
    """Temporarily set the net to eval mode."""
    training = net.training
    try:
        net.eval()
        yield net
    finally:
        if training:
            net.train()


def get_key_indices(input_space: gym.spaces.Dict) -> Tuple[Dict, Dict]:
    """Get index ranges that correspond to keys of a Dict space in flattened vector.

    Also returns a dictionary containing the shapes of these observation
    components.

    Returns:
        - A dictionary with keys corresponding to the observation components and
            values being tuples of the form (start, end), where start and end are
            the indices at which the observation component starts and ends. The
            nested dictionary structure of the observation is preserved.
        - A dictionary of the same structure but with values being the shapes
            of the observation components."""

    def _construct_dummy_obs(spaces_dict, counter=[0]):
        """Construct dummy observation which has an array repeating
        a different integer as the value of each component."""
        dummy_obs = {}
        for i, (k, v) in enumerate(spaces_dict.items()):
            if isinstance(v, gym.spaces.Dict):
                dummy_obs[k] = _construct_dummy_obs(v.spaces, counter)
            else:
                dummy_obs[k] = counter * np.ones(v.shape, dtype=np.int32)
                counter[0] += 1
        return dummy_obs

    dummy_obs = _construct_dummy_obs(input_space.spaces)
    flat_dummy_obs = gym.spaces.flatten(input_space, dummy_obs)

    def _get_indices_and_shape(dummy_obs, flat_dummy_obs):
        indices = {}
        shape = {}
        for k, v in dummy_obs.items():
            if isinstance(v, dict):
                indices[k], shape[k] = _get_indices_and_shape(v, flat_dummy_obs)
            else:
                where = np.where(flat_dummy_obs == v.flatten()[0])[0]
                indices[k] = (int(where[0]), int(where[-1]) + 1)
                shape[k] = v.shape
        return indices, shape

    return _get_indices_and_shape(dummy_obs, flat_dummy_obs)


def get_obs_indices(env: gym.Env) -> Tuple[Dict, Dict]:
    """Get index ranges that correspond to keys of the observation Dict space
    in flattened vector.

    Also returns a dictionary containing the shapes of these observation
    components.

    Args:
        env: The environment with Dict observation space.

    Returns:
        - A dictionary with keys corresponding to the observation components and
            values being tuples of the form (start, end), where start and end are
            the indices at which the observation component starts and ends. The
            nested dictionary structure of the observation is preserved.
        - A dictionary of the same structure but with values being the shapes
            of the observation components."""
    if isinstance(env, gym.vector.VectorEnv):
        single_env = env.env_fns[0]()
        single_obs_space = deepcopy(single_env.unwrapped.observation_space)
        single_env.close()
        del single_env
    else:
        single_obs_space = env.unwrapped.observation_space
    assert isinstance(
        single_obs_space, gym.spaces.Dict
    ), "Environment observation space must be of type Dict."
    return get_key_indices(single_obs_space)
