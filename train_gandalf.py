import transformers
import gandalf_dataset
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import peft
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import wandb
import hyper_params as h

wandb.login()

#configure distributed data parallel
is_ddp = int(os.environ.get("WORLD_SIZE", 1)) != 1
ds = gandalf_dataset.TrainDataset()
collator = transformers.DataCollatorForSeq2Seq(ds.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if is_ddp else None

#configure 4bit quantisation
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#create 4bit model
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", quantization_config=bnb_config, device_map=device_map)

#preprocessing for training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

#configure lora
config = LoraConfig(
    r=h.LORA_R, 
    lora_alpha=h.LORA_ALPHA, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=h.LORA_DROPOUT, 
    bias="none", 
    task_type="CAUSAL_LM"
)

#add lora adaptors to model
model = get_peft_model(model, config)

#deine training parameters
trainer = transformers.Trainer(
  model=model,
  train_dataset=ds,
  data_collator=collator,
  args=transformers.TrainingArguments(
    per_device_train_batch_size= h.TRAIN_BATCH_SIZE,
    num_train_epochs=h.TRAIN_NUM_EPOCHS,
    learning_rate= h.TRAIN_LEARNING_RATE,
    fp16=True,
    logging_steps= h.TRAIN_LOGGING_STEPS,
    optim="paged_adamw_8bit",
    evaluation_strategy="no",
    save_strategy="steps",
    eval_steps=None,
    save_steps= h.TRAIN_SAVE_STEPS,
    output_dir="./genius",
    save_total_limit= h.TRAIN_SAVE_TOTAL_LIMIT,
    report_to="wandb",
    ddp_find_unused_parameters=False if is_ddp else None,
  ),
)


model.config.use_cache = False
wandb.finish()
trainer.train()
model.save_pretrained("./weights")
