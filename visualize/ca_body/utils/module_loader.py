"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import copy
import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from attrdict import AttrDict

from torch import nn


logger: logging.Logger = logging.getLogger(__name__)


def load_module(
    module_name: str, class_name: Optional[str] = None, silent: bool = False
):
    """
    Load a module or class given the module/class name.

    Example:
    .. code-block:: python

        eye_geo = load_class("path.to.module", "ClassName")

    Args:
        module_name: str
        The full path of the module relative to the root directory. Ex: ``utils.module_loader``

        class_name: str
        The name of the class within the module to load.

        silent: bool
        If set to True, return None instead of raising an exception if module/class is missing

    Returns:
        object:
        The loaded module or class object.
    """
    try:
        module = importlib.import_module(f"visualize.{module_name}")
        if class_name:
            return getattr(module, class_name)
        else:
            return module
    except ModuleNotFoundError as e:
        if silent:
            return None
        logger.error(f"Module not found: {module_name}", exc_info=True)
        raise
    except AttributeError as e:
        if silent:
            return None
        logger.error(
            f"Can not locate class: {class_name} in {module_name}.", exc_info=True
        )
        raise


# pyre-ignore[3]
def make_module(mod_config: AttrDict, *args: Any, **kwargs: Any) -> Any:
    """
    A shortcut for making an object given the config and arguments

    Args:
        mod_config: AttrDict
        Config. Should contain keys: module_name, class_name, and optionally args

        *args
        Positional arguments.

        **kwargs
        Default keyword arguments. Overwritten by content from mod_config.args

    Returns:
        object:
        The loaded module or class object.
    """
    mod_config_dict = dict(mod_config)
    mod_args = mod_config_dict.pop("args", {})
    mod_args.update({k: v for k, v in kwargs.items() if k not in mod_args.keys()})
    mod_class = load_module(**mod_config_dict)
    return mod_class(*args, **mod_args)


def get_full_name(mod: object) -> str:
    """
    Returns a name of an object in a form <module>.<parent_scope>.<name>
    """
    mod_class = mod.__class__
    return f"{mod_class.__module__}.{mod_class.__qualname__}"


# pyre-fixme[3]: Return type must be annotated.
def load_class(class_name: str):
    """
    Load a class given the full class name.

    Example:
    .. code-block:: python

        class_instance = load_class("module.path.ClassName")

    Args:
        class_name: txt
        The full class name including the full path of the module relative to the root directory.
    Returns:
        A class
    """
    # This is a false-positive, pyre doesn't understand rsplit(..., 1) can only have 1-2 elements
    # pyre-fixme[6]: In call `load_module`, for 1st positional only parameter expected `bool` but got `str`.
    return load_module(*class_name.rsplit(".", 1))


@dataclass(frozen=True)
class ObjectSpec:
    """
    Args:
        class_name: str
        The full class name including the full path of the module relative to
        the root directory or just the name of the class within the module to
        load when module name is also provided.

        module_name: str
        The full path of the module relative to the root directory. Ex: ``utils.module_loader``

        kwargs: dict
        Keyword arguments for initializing the object.
    """

    class_name: str
    module_name: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


# pyre-fixme[3]: Return type must be annotated.
def load_object(spec: ObjectSpec, **kwargs: Any):
    """
    Instantiate an object given the class name and initialization arguments.

    Example:
    .. code-block:: python

        my_model = load_object(ObjectSpec(**my_model_config), in_channels=3)

    Args:
        spec: ObjectSpec
        An ObjectSpec object that specifies the class name and init arguments.

        kwargs: dict
        Additional keyword arguments for initialization.

    Returns:
        An object
    """
    if spec.module_name is None:
        object_class = load_class(spec.class_name)
    else:
        object_class = load_module(spec.module_name, spec.class_name)

    # Debug message for overriding the object spec
    for key in kwargs:
        if key in spec.kwargs:
            logger.debug(f"Overriding {key} as {kwargs[key]} in {spec}.")

    return object_class(**{**spec.kwargs, **kwargs})


# From DaaT merge. Fix here T145981161
# pyre-fixme[2]: parameter must be annotated.
# pyre-fixme[3]: Return type must be annotated.
def load_from_config(config: AttrDict, **kwargs):
    """Instantiate an object given a config and arguments."""
    assert "class_name" in config and "module_name" not in config
    config = copy.deepcopy(config)
    class_name = config.pop("class_name")
    object_class = load_class(class_name)
    return object_class(**config, **kwargs)


# From DaaT merge. Fix here T145981161
# pyre-fixme[2]: parameter must be annotated.
# pyre-fixme[3]: Return type must be annotated.
def forward_parameter_names(module):
    """Get the names arguments of the forward pass for the module.

    Args:
        module: a class with `forward()` method
    """
    names = []
    params = list(inspect.signature(module.forward).parameters.values())[1:]
    for p in params:
        if p.name in {"*args", "**kwargs"}:
            raise ValueError("*args and **kwargs are not supported")
        names.append(p.name)
    return names


# From DaaT merge. Fix here T145981161
def build_optimizer(config, model):
    """Build an optimizer given optimizer config and a model.

    Args:
        config: DictConfig
        model: nn.Module|Dict[str,nn.Module]

    """
    config = copy.deepcopy(config)

    if isinstance(model, nn.Module):
        if "per_module" in config:
            params = []
            for name, value in config.per_module.items():
                if not hasattr(model, name):
                    logger.warning(
                        f"model {model.__class__} does not have a submodule {name}, skipping"
                    )
                    continue

                params.append(
                    dict(
                        params=getattr(model, name).parameters(),
                        **value,
                    )
                )

            defined_names = set(config.per_module.keys())
            for name, module in model.named_children():
                n_params = len(list(module.named_parameters()))
                if name not in defined_names and n_params:
                    logger.warning(
                        f"not going to optimize module {name} which has {n_params} parameters"
                    )
            config.pop("per_module")
        else:
            params = model.parameters()
    else:
        # NOTE: can we do
        assert "per_module" in config
        assert isinstance(model, dict)
        for name, value in config.per_module.items():
            params = []
            for name, value in config.per_module.items():
                if name not in model:
                    logger.warning(f"not aware of {name}, skipping")
                    continue
                params.append(
                    dict(
                        params=model[name].parameters(),
                        **value,
                    )
                )

    return load_from_config(config, params=params)


# From DaaT merge. Fix here T145981161
class ForwardFilter:
    """A module that filters out arguments for the `forward()`."""

    # pyre-ignore
    def __init__(self, module, optional: bool = False) -> None:
        # pyre-ignore
        self.module = module
        # pyre-ignore
        self.input_names = set(forward_parameter_names(module))

    # pyre-ignore
    def __call__(self, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.input_names}
        return self.module(**filtered_kwargs)
