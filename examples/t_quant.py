from auto_round import AutoRound
 
# Load a model (supports FP8/BF16/FP16/FP32)
model_name_or_path = "Qwen/Qwen3-0.6B"
output_dir = "./Qwen3-0.6B-W4A16"
 
ar = AutoRound(model_name_or_path, scheme="W4A16", iters=0)
 
 
ar.quantize_and_save(output_dir=output_dir, format="llm_compressor")
