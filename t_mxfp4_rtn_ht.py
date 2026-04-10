from auto_round import AutoRound

# Load a model (supports FP8/BF16/FP16/FP32)
# model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
# output_dir = "./Llama-3.1-8B-Instruct_autoround_rtn_mxfp4_selective_ht"
model_name_or_path = "meta-llama/Llama-3.3-70B-Instruct"
output_dir = "./Llama-3.3-70B-Instruct_autoround_rtn_mxfp4_selective_ht"

ar = AutoRound(
    model_name_or_path,
    scheme="MXFP4",
    iters=0,
    hadamard_config="selective",
)


ar.quantize_and_save(output_dir=output_dir, format="auto_round")
