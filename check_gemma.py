import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from datasets import load_dataset

MODEL_ID = 'google/gemma-2-2b-it'

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
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

