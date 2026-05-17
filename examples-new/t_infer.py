from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./Qwen3-0.6B-quarot-default-mxfp4/Qwen3-0.6B-w4g128"
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
model.to("cuda")
print(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(inputs)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
