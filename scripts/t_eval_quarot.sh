#export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
NUM_GPUS=1
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

#MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
MODEL=../Qwen3-0.6B-quarot-online-mxfp4/Qwen3-0.6B-mxfp-w4g32/

MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.95,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False"


# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa,mmlu,hellaswag,gsm8k \
  --batch_size 8 \
  --limit 100 \
  --output_path lm_eval_results
