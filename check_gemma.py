import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import (
    PeftModel,
)
from datasets import load_dataset

# MODEL_ID = 'google/gemma-2-2b-it'
MODEL_ID = 'HuggingFaceTB/SmolLM-1.7B-Instruct'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
print(hasattr(model.config, 'final_logit_softcapping'))
# print(model.model.layers[0].self_attn.q_proj.lora_A.default.weight[:5, :5])
# print(model.model.layers[0].self_attn.q_proj.lora_B.default.weight[:5, :5])
# Gemma2ForCausalLM(
#   (model): Gemma2Model(
#     (embed_tokens): Embedding(256000, 2304, padding_idx=0)
#     (layers): ModuleList(
#       (0-25): 26 x Gemma2DecoderLayer(
#         (self_attn): Gemma2Attention(
#           (q_proj): lora.Linear(
#             (base_layer): Linear(in_features=2304, out_features=2048, bias=False)
#             (lora_dropout): ModuleDict(
#               (default): Identity()
#             )
#             (lora_A): ModuleDict(
# print(model)
# model = PeftModel.from_pretrained(
#     model, '/local_ssd2/nlyalyus/MODEL_DIR/gemma-2-2b-it/peft_init', trust_remote_code=True
# )
# print(model)
exit()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# print(model.device)
# print(model.model.device)
# standard
# input_text = "Write me a poem about Machine Learning."
# wikitext
# input_text = "It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 ."
# gsm8k
dataset = load_dataset('openai/gsm8k', 'main', split="train", trust_remote_code=True)
template = "Question:\n{question}\n\nAnswer:\n{answer}"
input_text = template.format(**dataset[6])
# template += '\n\nQuestion:\n{question}\n\nAnswer:\n'
# input_text = template.format(**dataset[7])
print(repr(input_text), '\n\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

# dolly
# data_item = {
#     "instruction": "When did Virgin Australia start operating?",
#     "response": "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.",
#     "instruction_2": "Which is a species of fish? Tope or Rope"
# }
# template = "Instruction:\n{instruction}\n\nResponse:\n{response}Instruction:\n{instruction_2}\n\nResponse:\n"
# # template = "{instruction}. {response}"
# input_text = template.format(**data_item)
# dataset = load_dataset('databricks/databricks-dolly-15k', split="train", trust_remote_code=True)
# template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
# input_text = template.format(**dataset[6])
# template += '\n\nInstruction:\n{instruction}\n\nResponse:\n'
# input_text = template.format(**dataset[7])
# print(repr(input_text))

# NOTE: standard
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=32)

# NOTE: chat template
# messages = [
#     {"role": "user", "content": input_text},
# ]
# input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
# outputs = model.generate(input_ids, max_new_tokens=32)

print(repr(tokenizer.decode(outputs[0])))

