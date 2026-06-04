export CUDA_VISIBLE_DEVICES=6
export VLLM_WORKER_MULTIPROC_METHOD=spawn
 
NUM_GPUS=1
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
 
#MODEL=./Llama-3.1-8B-Instruct_autoround_rtn_mxfp4
MODEL="Qwen/Qwen3-0.6B"
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False"
 
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa \
  --batch_size 8 \
  --output_path lm_eval_results
