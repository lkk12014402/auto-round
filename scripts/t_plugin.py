import auto_round.vllm_plugin  # 注册插件

from vllm import LLM, SamplingParams


if __name__ == '__main__':
    llm = LLM(
        model="../Qwen3-0.6B-quarot-online-mxfp4/Qwen3-0.6B-mxfp-w4g32/",
        quantization="spinquant_mxfp4",
        gpu_memory_utilization=0.8,
    )

    outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=64))
    for o in outputs:
        print(o.outputs[0].text)
