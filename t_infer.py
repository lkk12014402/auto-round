from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "./Llama-3.1-8B-Instruct_autoround_rtn_mxfp4_ht"
#model_name = "./Llama-3.1-8B-Instruct_autoround_iters200_mxfp4"
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
model_name = "./Llama-3.1-8B-Instruct_autoround_rtn_nvfp4"
model_name = "./Llama-3.1-8B-Instruct_autoround_rtn_nvfp4_ht"
model_name = "./Llama-3.1-8B-Instruct_autoround_rtn_nvfp4_ht_random"
model_name = "./Llama-3.1-8B-Instruct_autoround_rtn_mxfp4_selective_ht"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
model.to("cuda")
print(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(inputs)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
