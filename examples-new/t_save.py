from auto_round import AutoRound


model = "Qwen/Qwen3-0.6B"
output_dir = "Qwen3-0.6B-quarot-default-mxfp4"

# "quarot" applies R1+R2 with fixed Hadamard, then quantizes
#autoround = AutoRound(model, rotation_config="quarot", scheme="W4A16", iters=0)
#autoround = AutoRound(model, scheme="W4A16", iters=0)
def _run_autoround_impl(model_name, output_dir, device,
                        nsamples=128, seqlen=512, rotation_size=128,
                        online_r1=True):
    """Core auto-round runner (shared by online/offline variants).

    Uses AutoRound(rotation_config=SpinQuantConfig(...)) so rotation is
    applied automatically at Phase 4.5 before quantization.

    Args:
        online_r1: If True, R1 is applied as an online hook (default).
            If False, R1 is fused into weights offline (same as llm-compressor).
    """
    from auto_round import AutoRound
    from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

    rotation_config = None
    if True:
        rotation_config = SpinQuantConfig(
            r1=True, r2=True, r3=False, r4=False,
            #rotation_size=rotation_size,
            rotation_size=None,
            online_r1_rotation=online_r1,
            trainable_rotation=False,
            trainable_smooth=False,
        )

    ar = AutoRound(
        model_name,
        rotation_config=rotation_config,
        scheme="W4A16",
        iters=0,
        nsamples=nsamples,
        seqlen=seqlen,
        device_map=device,
    )

    ar.quantize_and_save(output_dir=output_dir, format="auto_round")


_run_autoround_impl(model, output_dir, device="auto", online_r1=False)
