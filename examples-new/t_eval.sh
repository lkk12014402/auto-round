#export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=5
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
NUM_GPUS=1
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

#MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
#MODEL=meta-llama/Llama-3.1-8B-Instruct
MODEL=Qwen/Qwen3-0.6B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,add_bos_token=True,softmax_dtype=float32,parallelize=True"

# GSM8k
lm_eval \
  --model hf \
  --model_args $MODEL_ARGS \
  --tasks piqa \
  --batch_size 8 \
  --gen_kwargs '{"max_gen_toks": 2048}' \
  --output_path lm_eval_results

exit


MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False"


# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa \
  --batch_size 8 \
  --output_path lm_eval_results
