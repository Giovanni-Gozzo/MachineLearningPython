from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

model_name = "lmsys/vicuna-13b-v1.5"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe= pipeline('text2text-generation', model=model, tokenizer=tokenizer)

result = pipe("translate English to French: Hello, my name is John", max_length=1000)
print(result)
