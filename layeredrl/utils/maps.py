from typing import Callable, List, Optional, Tuple

import torch


class RangesMap:
    """Keeps only specified index ranges of the input tensor."""

    def __init__(self, ranges: List[Tuple[Optional[int], Optional[int]]]):
        """
        Args:
            ranges: List of index ranges to keep. Each range is a tuple
                of the form (start, end) with start and end being integers or None
                (indicating no bound of the range)."""
        self.ranges = ranges

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor.
        Returns:
            Output tensor with only the specified ranges."""
        return torch.cat([x[..., start:end] for start, end in self.ranges], dim=-1)


class ConcatMap:
    """Concatenates multiple maps."""

    def __init__(self, maps: List[Callable]):
        """Args:
        maps: List of maps to concatenate."""
        self.maps = maps

    def __call__(self, *args) -> torch.Tensor:
        """Args:
            *args: Input tensors.
        Returns:
            Output tensor with the maps concatenated."""
        tensors = [map(v) for map, v in zip(self.maps, args)]
        return torch.cat(tensors, dim=-1)
