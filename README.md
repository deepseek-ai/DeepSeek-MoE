<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="images/logo.svg" width="60%" alt="DeepSeek LLM" />
</div>
<hr>
<div align="center">

  <a href="https://www.deepseek.com/" target="_blank">
    <img alt="Homepage" src="images/badge.svg" />
  </a>
  <a href="https://chat.deepseek.com/" target="_blank">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20LLM-536af5?color=536af5&logoColor=white" />
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" />
  </a>

</div>

<div align="center">

  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" />
  </a>
  <a href="images/qr.jpeg" target="_blank">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white" />
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" />
  </a>

</div>

<div align="center">

  <a href="LICENSE-CODE">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53">
  </a>
  <a href="LICENSE-MODEL">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53">
  </a>
</div>


<p align="center">
  <a href="#3-model-downloads">Model Download</a> |
  <a href="#2-evaluation-results">Evaluation Results</a> |
  <a href="#4-quick-start">Quick Start</a> |
  <a href="#5-license">License</a> |
  <a href="#6-citation">Citation</a>
</p>


## 1. Introduction

## 2. Evaluation Results

## 3. Model Downloads

We release the DeepSeek MoE 16B, including both base and chat models, to the public. To support a broader and more diverse range of research within both academic and commercial communities. Please **note** that the use of this model is subject to the terms outlined in [License section](#5-license). Commercial usage is permitted under these terms.

### Huggingface

|         Model         | Sequence Length |                                Download                                 |
|:---------------------:|:---------------:|:-----------------------------------------------------------------------:|
| DeepSeek MoE 16B Base  |      4096       | 🤗 [HuggingFace](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base)  |
| DeepSeek MoE 16B Chat  |      4096       | 🤗 [HuggingFace](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat)  |

## 4. Quick Start
### Installation

On the basis of `Python >= 3.8` environment, install the necessary dependencies by running the following command:

```shell
pip install -r requirements.txt
```

### Inference with Huggingface's Transformers

You can directly employ [Huggingface's Transformers](https://github.com/huggingface/transformers) for model inference.

**Text Completion**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

**Chat Completion**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/deepseek-moe-16b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

messages = [
    {"role": "user", "content": "Who are you?"}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)
```

Avoiding the use of the provided function `apply_chat_template`, you can also interact with our model following the sample template. Note that `messages` should be replaced by your input.

```
User: {messages[0]['content']}

Assistant: {messages[1]['content']}<｜end▁of▁sentence｜>User: {messages[2]['content']}

Assistant:
```

**Note:** By default (`add_special_tokens=True`), our tokenizer automatically adds a `bos_token` (`<｜begin▁of▁sentence｜>`) before the input text. Additionally, since the system prompt is not compatible with this version of our models, we DO NOT RECOMMEND including the system prompt in your input.

### How to Fine-tune DeepSeek-MoE

We provide script `fintune/finetune.py` for users to finetune our models on downstream tasks.

The script supports the training with [DeepSpeed](https://github.com/microsoft/DeepSpeed). You need install required packages by:

```bash
pip install -r requirements.txt
```

Please follow [Sample Dataset Format](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) to prepare your training data.
Each item has two required fields `instruction` and `output`.

After data preparation, you can use the sample shell script to finetune deepseek-MoE model. 
Remember to specify `DATA_PATH`, `OUTPUT_PATH`.
And please choose appropriate hyper-parameters(e.g., `learning_rate`, `per_device_train_batch_size`) according to your scenario.

```bash
DATA_PATH="<your_data_path>"
OUTPUT_PATH="<your_output_path>"
MODEL_PATH="<your_model_path>"

cd finetune
deepspeed finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True \
    --use_lora False
```

You can also finetune the model with 4/8-bits qlora, feel free to try it.
```bash
DATA_PATH="<your_data_path>"
OUTPUT_PATH="<your_output_path>"
MODEL_PATH="<your_model_path>"

cd finetune
deepspeed finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --bf16 True \
    --use_lora True \
    --bits 4 \
    --max_grad_norm 0.3 \
    --double_quant \
    --lora_r 64 \
    --lora_alpha 16 \
    --quant_type nf4 \
```

## 5. License
This code repository is licensed under the MIT License. The use of DeepSeek models is subject to the Model License. DeepSeek supports commercial use.

See the [LICENSE-CODE](LICENSE-CODE) and [LICENSE-MODEL](LICENSE-MODEL) for more details.

## 6. Citation


## 7. Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](mailto:service@deepseek.com).
