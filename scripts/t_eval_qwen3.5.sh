#export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=7
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
NUM_GPUS=2
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

#MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
MODEL=Qwen/Qwen3.5-2B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,add_bos_token=True,softmax_dtype=float32,parallelize=True"

# GSM8k
lm_eval \
  --model hf \
  --model_args $MODEL_ARGS \
  --tasks piqa,mmlu,hellaswag,gsm8k \
  --batch_size 8 \
  --gen_kwargs '{"max_gen_toks": 2048}' \
  --limit 100 \
  --output_path lm_eval_results


MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.95,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False"


# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa,mmlu,hellaswag,gsm8k \
  --batch_size 8 \
  --limit 100 \
  --output_path lm_eval_results
