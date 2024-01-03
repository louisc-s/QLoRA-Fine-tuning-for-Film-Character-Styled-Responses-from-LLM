
import transformers as t
import torch
import peft
import time

from bnb_config import bnb_config
from lora_config import lora_config

#define fine-tuned LLM model weights' location
peft_model_dir="./gandalf/checkpoint-300"

#define base LLM model and tokeniser
MODEL_NAME = "NousResearch/Llama-2-7b-hf"
tokenizer = t.AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token_id = 0

#load base LLM model
m = t.AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      quantization_config=bnb_config,
      use_cache=True,
      device_map= None
  )

#load fine-tuned LLM model weights
m = peft.PeftModel.from_pretrained(m, peft_model_dir)

device = "cuda"
m.to(device)

#define input prompt 
TEMPLATE = "Below is something that a person has said to you. Write a response to that person.\n\n### Line:\n{line}\n\n### Response:\n"
LINE = "Hello what is your name?"
prompt = TEMPLATE.format(line = LINE)

#generate output response 
pipe = t.pipeline(task="text-generation", model=m, tokenizer=tokenizer, max_length=500)
print("pipe(prompt)", pipe(prompt))




