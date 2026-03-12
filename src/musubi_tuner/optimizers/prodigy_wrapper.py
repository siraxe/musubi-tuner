"""
Wrapper for Prodigy optimizers to reduce state file size.

Prodigy optimizer stores full parameter copies in its state dict, causing
large optimizer.bin files. This wrapper filters the state dict to only
save essential data.
"""

from typing import Any, Dict, List, Optional, Union
import torch


class ProdigyStateFilter:
    """Mixin class to filter Prodigy's state_dict and reduce file size."""

    def state_dict(self, *args, **kwargs):
        """Override state_dict to filter out large unnecessary data."""
        orig_state_dict = super().state_dict(*args, **kwargs)

        # Filter each parameter's state to remove large cached data
        new_state = {}
        for param_id, param_state in orig_state_dict.get('state', {}).items():
            filtered_state = {}
            for key, value in param_state.items():
                # Keep only essential state keys, filter out large caches
                # Prodigy stores 'd0' (gradient exp moving avg) which is needed
                # But it also stores 'p0' (original parameters) which is large and can be reconstructed
                if key == 'p0':
                    # Skip storing original parameters - they can be reconstructed from current params
                    continue
                filtered_state[key] = value
            new_state[param_id] = filtered_state

        orig_state_dict['state'] = new_state
        return orig_state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Override load_state_dict to handle filtered state."""
        # First load the filtered state
        result = super().load_state_dict(state_dict)

        # Reconstruct p0 (initial parameters) for any state missing it
        # After load_state_dict, self.state maps param tensors directly to their state dicts
        # p0 is used by Prodigy to track distance from initialization
        # Note: Prodigy stores p0 as a sliced flattened tensor: p.flatten()[::slice_p]
        # Get slice_p from param_groups (default to 1 if not found)
        slice_p = self.param_groups[0].get('slice_p', 1) if self.param_groups else 1

        for param, param_state in self.state.items():
            if 'p0' not in param_state:
                # Use current param value as p0 (sliced flattened) - matches Prodigy's initialization
                param_state['p0'] = param.detach().flatten()[::slice_p].clone()

        return result


def wrap_prodigy_optimizer(optimizer_class):
    """
    Wrap a Prodigy optimizer class with state filtering.

    Args:
        optimizer_class: The Prodigy optimizer class to wrap

    Returns:
        A wrapped class with filtered state_dict
    """
    class WrappedProdigy(ProdigyStateFilter, optimizer_class):
        """Wrapped Prodigy optimizer with filtered state dict."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    # Preserve the original class name and module for serialization
    WrappedProdigy.__name__ = optimizer_class.__name__
    WrappedProdigy.__qualname__ = optimizer_class.__qualname__

    return WrappedProdigy


# Common Prodigy optimizer names that should be wrapped
PRODIGY_OPTIMIZERS = [
    'Prodigy',
    'ProdigyPlusScheduleFree',
]


def is_prodigy_optimizer(optimizer_name: str) -> bool:
    """Check if an optimizer name indicates a Prodigy variant."""
    optimizer_lower = optimizer_name.lower()
    return any(
        prodigy_name.lower() in optimizer_lower
        for prodigy_name in PRODIGY_OPTIMIZERS + ['prodigy']
    )


def get_prodigy_wrapper(optimizer_type: str, optimizer_module):
    """
    Get the appropriate Prodigy optimizer class, wrapped with state filtering.

    Args:
        optimizer_type: The name of the optimizer class
        optimizer_module: The module containing the optimizer

    Returns:
        Wrapped optimizer class or original class if not Prodigy
    """
    if is_prodigy_optimizer(optimizer_type):
        optimizer_class = getattr(optimizer_module, optimizer_type)
        return wrap_prodigy_optimizer(optimizer_class)
    return getattr(optimizer_module, optimizer_type)
