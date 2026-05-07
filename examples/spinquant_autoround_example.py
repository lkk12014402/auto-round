"""
Example: Using SpinQuant with Intel AutoRound.

This script demonstrates how to integrate SpinQuant preprocessing
with AutoRound quantization for improved low-bit LLM accuracy.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# Method 1: Manual integration (no source code changes to AutoRound)
# ==============================================================================

def example_manual_integration():
    """Run SpinQuant manually before AutoRound quantization."""
    
    from auto_round import AutoRound
    from auto_round.algorithms.transforms.spinquant import (
        SpinQuantConfig,
        SpinQuantPreprocessor,
    )
    from auto_round.calib_dataset import get_dataloader
    
    # 1. Load model
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Prepare calibration data for SpinQuant
    sq_dataloader = get_dataloader(
        tokenizer,
        seqlen=2048,
        dataset_name="NeelNanda/pile-10k",
        nsamples=128,
        bs=1,
    )
    
    # 3. Run SpinQuant preprocessing
    print("=" * 60)
    print("Step 1: SpinQuant Preprocessing")
    print("=" * 60)
    
    sq_config = SpinQuantConfig(
        r1=True,                    # Enable hidden_size rotation
        r2=True,                    # Enable head_dim rotation
        r3=False,                   # Skip online R3 (not needed for W4-only)
        r4=False,                   # Skip online R4 (not needed for W4-only)
        trainable_rotation=True,    # Learn optimal rotations
        iters=200,                  # Training iterations
        lr=1e-4,                    # Rotation learning rate
        smooth_lr=1e-3,             # Smooth value learning rate
        loss_type="kl_top",         # KL divergence on top-k logits
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    preprocessor = SpinQuantPreprocessor(model, sq_config)
    preprocessor.preprocess(sq_dataloader)
    
    # 4. Run AutoRound quantization
    print("=" * 60)
    print("Step 2: AutoRound Quantization")
    print("=" * 60)
    
    autoround = AutoRound(
        model,
        tokenizer,
        bits=4,
        group_size=128,
        sym=True,
        iters=200,
        lr=None,
        nsamples=128,
        seqlen=2048,
        batch_size=8,
    )
    
    # Quantize (model is already SpinQuant-rotated)
    autoround.quantize()
    
    # 5. Export
    autoround.save_quantized("./llama-1b-w4-spinquant")
    
    print("Done! Model saved to ./llama-1b-w4-spinquant")


# ==============================================================================
# Method 2: Integrated usage (with modified AutoRound source)
# ==============================================================================

def example_integrated():
    """Use SpinQuant through AutoRound's unified API."""
    
    from auto_round import AutoRound
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # AutoRound handles both SpinQuant and quantization
    autoround = AutoRound(
        model,
        tokenizer,
        bits=4,
        group_size=128,
        # === SpinQuant parameters ===
        enable_spinquant=True,
        spinquant_iters=200,
        spinquant_lr=1e-4,
        spinquant_r1=True,
        spinquant_r2=True,
        spinquant_trainable=True,
        # ============================
    )
    
    # SpinQuant preprocessing runs automatically before quantization
    autoround.quantize()
    autoround.save_quantized("./llama-1b-w4-spinquant-auto")


# ==============================================================================
# Method 3: Fixed Hadamard rotation (no training, faster)
# ==============================================================================

def example_fixed_hadamard():
    """Use fixed Hadamard rotation without training."""
    
    from auto_round import AutoRound
    from auto_round.algorithms.transforms.spinquant import (
        SpinQuantConfig,
        SpinQuantPreprocessor,
    )
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use fixed Hadamard (no training needed)
    sq_config = SpinQuantConfig(
        r1=True,
        r2=True,
        trainable_rotation=False,   # FIXED Hadamard, no training
        iters=0,
    )
    
    preprocessor = SpinQuantPreprocessor(model, sq_config)
    preprocessor.preprocess(dataloader=None)  # No dataloader needed
    
    # Continue with AutoRound
    autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
    autoround.quantize()
    autoround.save_quantized("./llama-1b-w4-hadamard")


# ==============================================================================
# Method 4: Advanced - Custom training loop with SpinQuant
# ==============================================================================

def example_custom_training():
    """Custom training loop with manual optimizer management."""
    
    import torch.nn.functional as F
    from auto_round.algorithms.transforms.spinquant import (
        SpinQuantConfig,
        SpinQuantPreprocessor,
        create_spinquant_optimizer,
        spinquant_loss_fn,
    )
    from auto_round.calib_dataset import get_dataloader
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Setup SpinQuant but don't train yet
    sq_config = SpinQuantConfig(
        r1=True,
        r2=True,
        trainable_rotation=True,
        iters=0,  # We'll handle training manually
    )
    
    preprocessor = SpinQuantPreprocessor(model, sq_config)
    # Only initialize rotations, don't train or fuse
    preprocessor._init_rotation_matrices()
    
    # Create custom optimizer
    optimizer = create_spinquant_optimizer(model, lr=1e-4, smooth_lr=1e-3)
    
    # Clone original model
    original_model = type(model).from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    original_model.eval()
    for p in original_model.parameters():
        p.requires_grad = False
    
    # Custom training loop
    dataloader = get_dataloader(tokenizer, seqlen=2048, dataset_name="NeelNanda/pile-10k", nsamples=128)
    model.train()
    
    for step, batch in enumerate(dataloader):
        if step >= 200:
            break
        
        # Rotated model forward
        outputs = model(**batch)
        logits = outputs.logits
        
        # Original model forward
        with torch.no_grad():
            ori_logits = original_model(**batch).logits
        
        # Loss
        loss = spinquant_loss_fn(logits, ori_logits, loss_type="kl_top", kl_top_k=1000)
        
        # Optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
    
    # Fuse rotations
    preprocessor._fuse_offline_rotations()
    preprocessor._cleanup()
    
    # Now model is ready for AutoRound
    from auto_round import AutoRound
    autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
    autoround.quantize()


if __name__ == "__main__":
    print("SpinQuant + AutoRound Integration Examples")
    print("=" * 60)
    print("Choose an example to run:")
    print("  1. Manual integration (recommended)")
    print("  2. Integrated API (requires source modifications)")
    print("  3. Fixed Hadamard (fastest, no training)")
    print("  4. Custom training loop")
    print("=" * 60)
    
    # Run example 1 by default
    example_manual_integration()

