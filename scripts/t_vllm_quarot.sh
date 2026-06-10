export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn


#NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
NUM_GPUS=2
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.4}

#MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
#MODEL=./selective_rotation_mxfp4_model/Qwen3-0.6B-mxfp-w4g32/
MODEL=rotation_mxfp4_model_8B/Qwen3-8B-mxfp-w4g32/
#MODEL=selective_rotation_mxfp4_model_8B/Qwen3-8B-mxfp-w4g32/
#MODEL=./Llama-3.1-8B-Instruct_autoround_iters200_mxfp4
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=16384,max_num_seqs=32,add_bos_token=True,gpu_memory_utilization=0.3,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False"


# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks piqa \
  --batch_size 8 \
  --output_path lm_eval_results
