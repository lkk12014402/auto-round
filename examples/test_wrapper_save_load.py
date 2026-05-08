#!/usr/bin/env python3
"""Test InputRotationWrapperHadamard: forward equivalence and save/load roundtrip.

Tests:
1. Wrapper forward == original forward (rotation cancels out)
2. Model save_pretrained + from_pretrained preserves accuracy
3. state_dict key names are compatible (no .original_module. prefix)
"""

import os
import sys
import tempfile
import logging

import torch
import torch.nn as nn

# Ensure auto_round is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    InputRotationWrapperHadamard,
    get_hadamard_K,
    matmul_hadU,
)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger("test_wrapper")


def test_wrapper_forward_equivalence():
    """Test that wrapper(x) == original(H(x)) where weights are H-rotated."""
    logger.info("=== Test 1: Forward equivalence ===")

    in_features = 128
    out_features = 64
    batch = 4
    seq_len = 16

    torch.manual_seed(42)
    linear = nn.Linear(in_features, out_features, bias=True)
    x = torch.randn(batch, seq_len, in_features)

    # Save original output
    with torch.no_grad():
        y_original = linear(x)

    # Create a fresh linear with same weights, rotate it, and wrap
    linear_rotated = nn.Linear(in_features, out_features, bias=True)
    linear_rotated.load_state_dict(linear.state_dict())

    # Rotate weights: W' = matmul_hadU(W)
    hadamard_K, K = get_hadamard_K(in_features)
    with torch.no_grad():
        linear_rotated.weight.data = matmul_hadU(
            linear_rotated.weight.data, hadamard_K=hadamard_K, K=K
        )

    # Wrap with InputRotationWrapperHadamard
    wrapper = InputRotationWrapperHadamard(
        original_module=linear_rotated,
        rotation_size=in_features,
        hadamard_K=hadamard_K,
        K=K,
    )

    with torch.no_grad():
        y_wrapper = wrapper(x)

    cos_sim = torch.nn.functional.cosine_similarity(
        y_original.flatten(), y_wrapper.flatten(), dim=0
    ).item()
    max_diff = (y_original - y_wrapper).abs().max().item()

    logger.info(f"  cosine_similarity: {cos_sim:.8f}")
    logger.info(f"  max_abs_diff:      {max_diff:.2e}")

    assert cos_sim > 0.9999, f"cosine_similarity too low: {cos_sim}"
    assert max_diff < 1e-4, f"max_abs_diff too large: {max_diff}"
    logger.info("  PASSED ✓")


def test_wrapper_block_rotation():
    """Test wrapper with block rotation (rotation_size < in_features)."""
    logger.info("=== Test 2: Block rotation forward equivalence ===")

    in_features = 256
    out_features = 64
    rotation_size = 128

    torch.manual_seed(42)
    linear = nn.Linear(in_features, out_features, bias=False)
    x = torch.randn(2, 8, in_features)

    with torch.no_grad():
        y_original = linear(x)

    # Rotate weights with block rotation
    linear_rotated = nn.Linear(in_features, out_features, bias=False)
    linear_rotated.load_state_dict(linear.state_dict())

    hadamard_K, K = get_hadamard_K(rotation_size)
    # Build full block rotation matrix
    R = hadamard_K.to(torch.float64)
    if R.shape[0] != rotation_size:
        from auto_round.algorithms.transforms.spinquant.rotation_utils import get_hadamard_K as _ghK
        had_1, _ = _ghK(rotation_size // K)
        R = torch.kron(hadamard_K.to(torch.float64), had_1.to(torch.float64))

    with torch.no_grad():
        # Block-rotate in_channels: reshape weight and multiply each block
        W = linear_rotated.weight.data.to(torch.float64)
        # Weight shape: [out_features, in_features]
        # Block rotation on input channels
        n_blocks = in_features // rotation_size
        W_reshaped = W.reshape(out_features, n_blocks, rotation_size)
        R_f = R.to(W.device)
        W_rotated = torch.matmul(W_reshaped, R_f.t()).reshape(out_features, in_features)
        linear_rotated.weight.data = W_rotated.to(linear.weight.dtype)

    wrapper = InputRotationWrapperHadamard(
        original_module=linear_rotated,
        rotation_size=rotation_size,
        hadamard_K=hadamard_K,
        K=K,
    )

    with torch.no_grad():
        y_wrapper = wrapper(x)

    cos_sim = torch.nn.functional.cosine_similarity(
        y_original.flatten(), y_wrapper.flatten(), dim=0
    ).item()
    max_diff = (y_original - y_wrapper).abs().max().item()

    logger.info(f"  cosine_similarity: {cos_sim:.8f}")
    logger.info(f"  max_abs_diff:      {max_diff:.2e}")

    assert cos_sim > 0.9999, f"cosine_similarity too low: {cos_sim}"
    logger.info("  PASSED ✓")


def test_state_dict_key_names():
    """Test that state_dict() removes .original_module. prefix."""
    logger.info("=== Test 3: state_dict key compatibility ===")

    linear = nn.Linear(128, 64, bias=True)
    hadamard_K, K = get_hadamard_K(128)
    wrapper = InputRotationWrapperHadamard(linear, 128, hadamard_K, K)

    sd = wrapper.state_dict()
    keys = list(sd.keys())
    logger.info(f"  state_dict keys: {keys}")

    # Should NOT contain .original_module.
    for k in keys:
        assert ".original_module." not in k, f"Key has .original_module. prefix: {k}"

    # Should contain weight and bias (without prefix since we call with default prefix="")
    assert "weight" in keys, f"'weight' not found in {keys}"
    assert "bias" in keys, f"'bias' not found in {keys}"
    # Should contain hadamard_K buffer
    assert "hadamard_K" in keys, f"'hadamard_K' not found in {keys}"

    logger.info("  PASSED ✓")


def test_wrapper_in_model_state_dict():
    """Test that wrapper inside a model produces correct prefixed keys."""
    logger.info("=== Test 4: Wrapper in model state_dict ===")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(128, 64)

        def forward(self, x):
            return self.layer(x)

    model = SimpleModel()
    hadamard_K, K = get_hadamard_K(128)

    # Wrap the layer
    wrapper = InputRotationWrapperHadamard(model.layer, 128, hadamard_K, K)
    model.layer = wrapper

    sd = model.state_dict()
    keys = sorted(sd.keys())
    logger.info(f"  Model state_dict keys: {keys}")

    # Keys should be: layer.weight, layer.bias, layer.hadamard_K
    # NOT: layer.original_module.weight
    expected = {"layer.bias", "layer.hadamard_K", "layer.weight"}
    assert set(keys) == expected, f"Unexpected keys: {set(keys)} vs expected {expected}"

    logger.info("  PASSED ✓")


def test_load_from_state_dict():
    """Test that a wrapper can load state_dict saved without .original_module. prefix."""
    logger.info("=== Test 5: Load from state_dict ===")

    torch.manual_seed(42)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(128, 64)

        def forward(self, x):
            return self.layer(x)

    model = SimpleModel()
    hadamard_K, K = get_hadamard_K(128)

    # Rotate and wrap
    with torch.no_grad():
        model.layer.weight.data = matmul_hadU(model.layer.weight.data, hadamard_K=hadamard_K, K=K)
    wrapper = InputRotationWrapperHadamard(model.layer, 128, hadamard_K, K)
    model.layer = wrapper

    # Get reference output
    x = torch.randn(2, 8, 128)
    with torch.no_grad():
        y_ref = model(x)

    # Save state dict
    sd = model.state_dict()

    # Create a new model with wrapper and load
    model2 = SimpleModel()
    wrapper2 = InputRotationWrapperHadamard(model2.layer, 128, hadamard_K, K)
    model2.layer = wrapper2
    model2.load_state_dict(sd)

    with torch.no_grad():
        y_loaded = model2(x)

    max_diff = (y_ref - y_loaded).abs().max().item()
    logger.info(f"  max_abs_diff after load: {max_diff:.2e}")
    assert max_diff < 1e-6, f"Output changed after load: {max_diff}"

    logger.info("  PASSED ✓")


def test_save_load_with_transformers():
    """Test save_pretrained / from_pretrained with Qwen3-0.6B + online R1."""
    logger.info("=== Test 6: Full model save/load with transformers ===")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.info("  SKIPPED (transformers not installed)")
        return

    model_name = "Qwen/Qwen3-0.6B"
    logger.info(f"  Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Apply online R1
    from auto_round.algorithms.transforms.spinquant.preprocessor import (
        SpinQuantConfig,
        SpinQuantPreprocessor,
    )

    config = SpinQuantConfig(
        r1=True,
        r2=False,
        r3=False,
        r4=False,
        online_r1_rotation=True,
        trainable_rotation=False,
        trainable_smooth=False,
    )
    preprocessor = SpinQuantPreprocessor(model, config)
    preprocessor.preprocess()

    # Verify wrappers exist
    wrapper_count = 0
    for name, module in model.named_modules():
        if isinstance(module, InputRotationWrapperHadamard):
            wrapper_count += 1
    logger.info(f"  Found {wrapper_count} InputRotationWrapperHadamard wrappers")
    assert wrapper_count > 0, "No wrappers found after online R1"

    # Get reference output
    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        y_ref = model(**inputs).logits

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"  Saving to {tmpdir}...")
        model.save_pretrained(tmpdir)
        tokenizer.save_pretrained(tmpdir)

        # Check saved files
        import json
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path) as f:
            saved_config = json.load(f)
        logger.info(f"  Saved model type: {saved_config.get('model_type')}")

        # Check state dict keys - should not have .original_module.
        import safetensors.torch
        sd_files = [f for f in os.listdir(tmpdir) if f.endswith(".safetensors")]
        logger.info(f"  Safetensors files: {sd_files}")

        if sd_files:
            sd_path = os.path.join(tmpdir, sd_files[0])
            saved_keys = list(safetensors.torch.load_file(sd_path).keys())
            bad_keys = [k for k in saved_keys if ".original_module." in k]
            if bad_keys:
                logger.warning(f"  WARNING: Found .original_module. keys: {bad_keys[:5]}...")
            else:
                logger.info("  ✓ No .original_module. keys in saved state_dict")

        # Load back
        logger.info("  Loading saved model...")
        # For now, we load the base model and re-apply wrappers
        # (Full auto-reconstruction would require custom model class registration)
        model2 = AutoModelForCausalLM.from_pretrained(
            tmpdir,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        # Re-apply wrappers (needed until we register custom model class)
        # The weights are already rotated in the saved model, so we just
        # need to wrap the target modules with the same Hadamard
        hidden_size = model2.config.hidden_size
        rotation_size = hidden_size  # default
        hadamard_K, K = get_hadamard_K(rotation_size)

        layers = model2.model.layers if hasattr(model2, "model") else model2.transformer.h
        for layer in layers:
            attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attn
            mlp = layer.mlp

            for proj_name in ("q_proj", "k_proj", "v_proj"):
                if hasattr(attn, proj_name):
                    module = getattr(attn, proj_name)
                    w = InputRotationWrapperHadamard(module, rotation_size, hadamard_K, K)
                    setattr(attn, proj_name, w)
            for proj_name in ("gate_proj", "up_proj"):
                if hasattr(mlp, proj_name):
                    module = getattr(mlp, proj_name)
                    w = InputRotationWrapperHadamard(module, rotation_size, hadamard_K, K)
                    setattr(mlp, proj_name, w)

        with torch.no_grad():
            y_loaded = model2(**inputs).logits

        cos_sim = torch.nn.functional.cosine_similarity(
            y_ref.flatten(), y_loaded.flatten(), dim=0
        ).item()
        max_diff = (y_ref - y_loaded).abs().max().item()
        logger.info(f"  cosine_similarity: {cos_sim:.8f}")
        logger.info(f"  max_abs_diff:      {max_diff:.2e}")

        assert cos_sim > 0.999, f"cosine_similarity too low after load: {cos_sim}"
        logger.info("  PASSED ✓")


def test_proxy_properties():
    """Test that wrapper proxies weight, bias, in_features, out_features."""
    logger.info("=== Test 7: Proxy properties ===")

    linear = nn.Linear(128, 64, bias=True)
    hadamard_K, K = get_hadamard_K(128)
    wrapper = InputRotationWrapperHadamard(linear, 128, hadamard_K, K)

    assert wrapper.in_features == 128
    assert wrapper.out_features == 64
    assert wrapper.weight is linear.weight
    assert wrapper.bias is linear.bias

    logger.info("  PASSED ✓")


if __name__ == "__main__":
    test_wrapper_forward_equivalence()
    test_wrapper_block_rotation()
    test_state_dict_key_names()
    test_wrapper_in_model_state_dict()
    test_load_from_state_dict()
    test_proxy_properties()

    # Run the full model test only if explicitly requested
    if "--full" in sys.argv:
        test_save_load_with_transformers()
    else:
        logger.info("\n=== Skipping full model test (pass --full to run) ===")

    logger.info("\n✅ All tests passed!")
