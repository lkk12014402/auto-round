from typing import Dict, Any, Callable, Type

import torch
import torch.nn as nn

from .hooks import build_transform_hook, TransformHook


def _default_layer_filter(module: nn.Module) -> bool:
    """
    决定哪些层需要应用 transform。
    这里先简单示例：只对 Linear 生效，你可以按需要扩展规则。
    """
    return isinstance(module, nn.Linear)


def hadamard_prepare_for_quant(
    model: nn.Module,
    transform_cfg: Dict[str, Any],
    layer_filter: Callable[[nn.Module], bool] = _default_layer_filter,
) -> nn.Module:
    """
    在量化前对 model 做一次处理：
    - 对满足 layer_filter 的层，挂上 transform_hook & transform_state；
    - 其他逻辑交给 AutoRound 原有实现。

    使用方式：
        model = hadamard_prepare_for_quant(model, {
            "hook_name": "hadamard_mxfp4",
            "group_size": 32,
        })
    """
    hook = build_transform_hook(transform_cfg)
    if hook is None:
        return model

    # 注意：所有层可以共享一个 hook 实例，也可以 clone；这里直接共享即可
    for name, module in model.named_modules():
        if layer_filter(module):
            # 挂 hook 和一个状态 dict
            setattr(module, "transform_hook", hook)
            setattr(module, "transform_state", {})

    return model
