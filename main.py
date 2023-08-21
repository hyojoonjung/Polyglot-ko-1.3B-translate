import pandas as pd
#데이터셋을 입력하세요
df = pd.read_csv('data/train.csv')
df.head()
df['num'] = df.index
def makedata(x):
    if (x['num'] % 2 == 0):
        return f"### 한국어: {x['ko']}</끝>\n### 영어: {x['en']}</끝>"
    else:
        return f"### 영어: {x['en']}</끝>\n### 한국어: {x['ko']}</끝>"
data = []
data = pd.DataFrame(data)
data['text'] = df.apply(makedata, axis=1)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_id = "EleutherAI/polyglot-ko-1.3b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
model.device
# model = AutoModelForCausalLM.from_pretrained(model_id)
data = data.apply(lambda samples: tokenizer(samples["text"]), axis=1)

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

tokenNum_korean = 8611
tokenNum_english = 3029
tokenNum_colon = 29

import transformers
from transformers import Trainer
import numpy as np

class maskTrainer(Trainer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def compute_loss(self, model, inputs, return_outputs=False):
    if "token_type_ids" in inputs:
      del inputs["token_type_ids"]
    # print(inputs['labels'][1])
    for x in range(len(inputs['labels'])):
      if (inputs['labels'][x][3] == tokenNum_korean):
          maskindex = (inputs['labels'][x]==tokenNum_english).nonzero()[:, 0]
          temp = 0
          for i, index in enumerate(maskindex):
            if (inputs['labels'][x][index+1] != tokenNum_colon):
              maskindex = np.delete(maskindex.cpu(), i-temp)
              temp += 1

      elif (inputs['labels'][x][3] == tokenNum_english):
          maskindex = (inputs['labels'][x]==tokenNum_korean).nonzero()[:, 0]
          temp = 0
          for i, index in enumerate(maskindex):
            if (inputs['labels'][x][index+1] != tokenNum_colon):
              maskindex = np.delete(maskindex.cpu(), i-temp)
              temp += 1

      inputs['labels'][x][:maskindex[0]+2] = -100

    # print(inputs['labels'][1])
    outputs = model(**inputs)

    loss = outputs['loss']

    return (loss,outputs) if return_outputs else loss

# import transformers

tokenizer.pad_token = tokenizer.eos_token

trainer = maskTrainer(
    model=model,
    train_dataset=data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        fp16=True,
        output_dir="outputs",
        save_total_limit=2,
        logging_steps=300,
        report_to=["tensorboard"],
        num_train_epochs = 1,
        learning_rate=3e-4,
        resume_from_checkpoint=True,
        lr_scheduler_type= "cosine",
        #optim="paged_adamw_8bit"

    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
# trainer.train(resume_from_checkpoint=True)
trainer.train()

model.eval()
model.config.use_cache = True  # silence the warnings. Please re-enable for inference!

model.save_pretrained("./saved/translation/1.3B/")