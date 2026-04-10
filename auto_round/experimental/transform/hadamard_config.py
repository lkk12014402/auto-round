# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel, Field, field_validator

__all__ = ["HadamardConfig"]


class HadamardConfig(BaseModel):
    """
    Configuration of transforms to be applied to a model. This config is to be
    serialized within a model's `config.json` file
    """

    block_size: int = Field(default=32)

    hadamard_type: str = Field(default="hadamard")

    selector: Literal["all", "heuristic"] = Field(default="all")

    kurtosis_threshold: float = Field(default=5.0)

    energy_ratio_threshold: float = Field(default=0.05)

    score_threshold: float = Field(default=1.5)

    min_qk_head_dim: int = Field(default=64)

    # for random hadamard transform
    random_seed: bool = Field(default=False, exclude=True)

    @field_validator("hadamard_type")
    @classmethod
    def validate_hadamard_type(cls, v: str) -> str:
        allowed = {"hadamard", "random_hadamard"}
        if v not in allowed:
            raise ValueError(f"Unsupported hadamard_type: {v}. Supported values: {sorted(allowed)}")
        return v

    def to_transform_dict(self) -> dict:
        return {
            "block_size": self.block_size,
            "hadamard_type": self.hadamard_type,
        }
