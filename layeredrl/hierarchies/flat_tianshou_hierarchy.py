from .hierarchy import Hierarchy
from ..levels import TianshouLevel


class FlatTianshouHierarchy(Hierarchy):
    """A hierarchy consisting of a single level with a Tianshou policy."""

    def __init__(self, env, tianshou_config, device, **kwargs):
        """Initialize the hierarchy.

        Args:
            env: The environment.
            tianshou_config: The configuration of the Tianshou level.
            device: The device to use.
        """
        levels = [
            TianshouLevel(tianshou_config=tianshou_config, device=device, **kwargs)
        ]
        super().__init__(levels=levels, env=env, device=device)
