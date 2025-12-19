from .hierarchy import Hierarchy
from ..levels import RandomLevel


class RandomHierarchy(Hierarchy):
    """A hierarchy consisting of a single level returning random actions.

    The actions are sampled uniformly from the action space of the environment
    if the action space is finite/a finite interval."""

    def __init__(self, env, device):
        levels = [RandomLevel(device=device)]
        super().__init__(levels=levels, env=env, device=device)
