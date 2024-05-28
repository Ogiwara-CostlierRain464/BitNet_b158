import torch
from datasets import load_dataset, DatasetDict
from simple.bitnet158.llama import BitLlamaConfig, BitLlamaForCausalLM
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoTokenizer

ds_name = "range3/wiki40b-ja"
ds_train = load_dataset(ds_name, split="train")
ds_valid = load_dataset(ds_name, split="validation")

raw_datasets = DatasetDict(
    {
        "train": ds_train,
        "valid": ds_valid,
    }
)

context_length = 128
tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Swallow-7b-hf")

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)

config = BitLlamaConfig(
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    hidden_size=768,                  # BitNet論文より
    max_position_embeddings=1024,
    intermediate_size=1536, # Transformerの論文より、途中で拡大する。
    num_attention_heads=12,         # BitNet論文より
    num_hidden_layers=12,            # BitNet論文より
    num_key_value_heads=4,
    torch_dtype=torch.bfloat16,
    rms_norm_eps=1e-06,
)

model = BitLlamaForCausalLM(config)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

model_size = sum(t.numel() for t in model.parameters())
print(f"model size: {model_size/1000**2:.1f}M parameters")

args = TrainingArguments(
    output_dir="bitnet-127M",
    per_device_train_batch_size=160,
    per_device_eval_batch_size=160,
    evaluation_strategy="steps",
    eval_steps=5000,
    logging_steps=2000,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    weight_decay=0.1, # b158
    warmup_steps=5000,
    lr_scheduler_type="linear",  # BitNet論文より
    learning_rate=2.4e-3,  # BitNetb1論文より
    save_steps=2000,
    bf16=True,
    adam_beta1=0.9,  # BitNet論文より
    adam_beta2=0.95,  # BitNet論文より
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train(resume_from_checkpoint=True)

trainer.save_model()



