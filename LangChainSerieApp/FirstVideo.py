import os
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Template de prompt
template = "Write a {adjective} poem about {topic}."
prompt = PromptTemplate(input_variables=["adjective", "topic"], template=template)

# Création du prompt formaté
formatted_prompt = prompt.format(adjective="beautiful", topic="nature")

from transformers import pipeline, AutoTokenizer

# Charger le tokenizer explicitement
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

# Charger le modèle avec le pipeline en utilisant le tokenizer
pipe = pipeline("text-generation", model="microsoft/Phi-3.5-mini-instruct", tokenizer=tokenizer, trust_remote_code=True)

# Génération du texte basé sur le prompt formaté
generated_text = pipe(formatted_prompt)

# Affichage du résultat
print(generated_text)


# Génération du texte basé sur le prompt formaté
generated_text = pipe(formatted_prompt)

# Affichage du résultat
print(generated_text)



