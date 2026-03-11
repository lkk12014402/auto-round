#export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=5
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export VLLM_ENABLE_AR_EXT=1
export VLLM_AR_MXFP4_MODULAR_MOE=1
export VLLM_MXFP4_PRE_UNPACK_TO_FP8=1
export VLLM_MXFP4_PRE_UNPACK_WEIGHTS=0
export VLLM_ENABLE_STATIC_MOE=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1

#NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
NUM_GPUS=1
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

#MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
#MODEL=./Llama-3.1-8B-Instruct_autoround_rtn_mxfp4
MODEL=./Llama-3.1-8B-Instruct_autoround_iters200_mxfp4
MODEL=/models/Llama-3.1-8B-Instruct
MODEL_ARGS="pretrained=${MODEL},tensor_parallel_size=${NUM_GPUS},max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,add_bos_token=True,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048,enable_prefix_caching=False"


# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks gsm8k \
  --batch_size 8 \
  --limit 100 \
  --output_path lm_eval_results

exit

# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks hellaswag \
  --batch_size 8 \
  --output_path lm_eval_results


# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks mmlu_llama \
  --batch_size 8 \
  --output_path lm_eval_results

# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks gsm8k_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size 8 \
  --output_path lm_eval_results


exit

# Winogrande
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks winogrande \
  --num_fewshot=5 \
  --batch_size auto \
  --output_path lm_eval_results

# Hellaswag
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks hellaswag \
  --num_fewshot=10 \
  --batch_size auto \
  --output_path lm_eval_results

# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks gsm8k_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto \
  --output_path lm_eval_results

# MMLU-CoT 
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks mmlu_cot_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto \
  --output_path lm_eval_results
