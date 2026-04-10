#export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
NUM_GPUS=1
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

#MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
#MODEL=llm-compressor/examples/transform/data-quip-w4a16/
# MODEL=./Llama-3.1-8B-Instruct_autoround_rtn_mxfp4_ht
MODEL=./Llama-3.1-8B-Instruct_autoround_rtn_mxfp4_selective_ht
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16"


lm_eval \
  --model hf \
  --model_args $MODEL_ARGS \
  --tasks piqa \
  --batch_size 8 \
  --output_path lm_eval_results

exit

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
