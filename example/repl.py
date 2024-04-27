import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mybitnet import BitLlamaConfig

model_name = "HachiML/myBit-Llama2-jp-127M-4"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BitLlamaConfig()
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)

prompt = "ずんだもんとは、"
input_ids = tokenizer.encode(
    prompt,
    return_tensors="pt"
)

tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=128,
)

out = tokenizer.decode(tokens[0], skip_special_tokens=True)
print(out)