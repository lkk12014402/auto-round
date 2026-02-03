import torch
from typing import Optional, Dict, Any

from .transforms import build_transform  # 你已有的 TRANSFORMS 注册
from ..experimental.triton.mxfp4 import mxfp4_forward_kernel_wrapper


class TransformHook:
    """
    抽象基类：定义 AutoRound 支持的可插拔 transform 接口。
    """

    def pre_quant_weight(
        self,
        weight: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        extra_state: Dict[str, Any],
    ) -> torch.Tensor:
        """
        权重量化前调用，可返回变换后的 weight。
        extra_state 可以记录中间状态，用于 post_quant_weight。
        """
        return weight

    def post_quant_weight(
        self,
        orig_layer: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        extra_state: Dict[str, Any],
    ) -> None:
        """
        unwrapper 时调用，用于把需要的参数/矩阵挂到 orig_layer 上。
        """
        pass

    def pre_quant_act(
        self,
        x: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        extra_state: Dict[str, Any],
    ) -> torch.Tensor:
        """
        activation 量化前调用。
        """
        return x

    def infer_input(
        self,
        x: torch.Tensor,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """
        推理侧输入处理。默认直接调用 layer.qdq_input(x)。
        """
        if hasattr(layer, "qdq_input"):
            return layer.qdq_input(x)
        return x


class HadamardMXFP4Hook(TransformHook):
    """
    针对 hadamard + mxfp4 的 Hook：
    - 量化阶段：对权重/激活做 Hadamard 变换；
    - unwrapper: 生成且挂载 forward_hadamard_matrix；
    - 推理阶段：用 Triton mxfp4 kernel 在 Hadamard 域 qdq 输入。
    """

    def __init__(self, group_size: int = 32):
        self.group_size = group_size
        # 复用你已有的 HadamardTransform
        self._transform = build_transform("hadamard", group_size=group_size)

    # -------- 训练/量化阶段 --------

    def pre_quant_weight(
        self,
        weight: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        extra_state: Dict[str, Any],
    ) -> torch.Tensor:
        weight = self._transform(weight.to(device))
        extra_state["group_size"] = self.group_size
        return weight

    def post_quant_weight(
        self,
        orig_layer: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        extra_state: Dict[str, Any],
    ) -> None:
        # 生成并挂载矩阵（与当前 unwrapper 里那段逻辑等价）
        transform_matrix = self._transform.get_transform_matrix(device, dtype).cpu()
        orig_layer.transform_config = {
            "transform_class": "hadamard",
            "group_size": self.group_size,
        }
        orig_layer.forward_hadamard_matrix = transform_matrix

    def pre_quant_act(
        self,
        x: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        extra_state: Dict[str, Any],
    ) -> torch.Tensor:
        return self._transform(x.to(device))

    # -------- 推理阶段 --------

    def infer_input(
        self,
        x: torch.Tensor,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """
        在 MXFP4 qmodule.forward 中调用：
        - 如果有 forward_hadamard_matrix 就用 mxfp4 Triton kernel；
        - 否则 fallback 到普通 qdq_input。
        """
        if not hasattr(layer, "forward_hadamard_matrix"):
            if hasattr(layer, "qdq_input"):
                return layer.qdq_input(x)
            return x

        orig_shape = x.shape
        x_flat = x.contiguous().flatten(end_dim=-2)
        qdq_input, _ = mxfp4_forward_kernel_wrapper(
            x_flat,
            layer.forward_hadamard_matrix,
        )
        return qdq_input.reshape(orig_shape)


HOOKS = {
    "hadamard_mxfp4": HadamardMXFP4Hook,
}


def build_transform_hook(config: Optional[dict]):
    """
    根据配置构造 hook。
    config 示例:
        {
            "hook_name": "hadamard_mxfp4",
            "group_size": 32,
        }
    """
    if not config:
        return None
    hook_name = config.get("hook_name")
    if not hook_name:
        return None
    hook_cls = HOOKS.get(hook_name)
    if hook_cls is None:
        raise ValueError(f"Unknown transform hook: {hook_name}")
    kwargs = {k: v for k, v in config.items() if k != "hook_name"}
    return hook_cls(**kwargs)
