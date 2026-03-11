#export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=6
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
NUM_GPUS=1
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

#MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
MODEL=./Llama-3.1-8B-Instruct_autoround_rtn_mxfp4
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16"

lm_eval \
  --model hf \
  --model_args $MODEL_ARGS \
  --tasks gsm8k \
  --batch_size 8 \
  --limit 100 \
  --output_path lm_eval_results

exit


# GSM8k
lm_eval \
  --model hf \
  --model_args $MODEL_ARGS \
  --tasks piqa \
  --batch_size 8 \
  --output_path lm_eval_results


# GSM8k
lm_eval \
  --model hf \
  --model_args $MODEL_ARGS \
  --tasks hellaswag \
  --batch_size 8 \
  --output_path lm_eval_results


# GSM8k
lm_eval \
  --model hf \
  --model_args $MODEL_ARGS \
  --tasks mmlu_llama \
  --batch_size 8 \
  --output_path lm_eval_results

# GSM8k
lm_eval \
  --model hf \
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
